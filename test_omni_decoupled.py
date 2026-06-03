"""Decoupled omni e2e test: thinker subprocess → file → talker subprocess → WAV.

Runs `omni_decoupled.thinker_server` on cuda:0 and `omni_decoupled.talker_client`
on cuda:1 (or the same GPU if only one is visible). Verifies the WAV is non-empty.

Usage:
  CUDA_VISIBLE_DEVICES=5,6 python test_omni_decoupled.py
"""
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_omni_decoupled")

PROMPT = os.environ.get("OMNI_TEST_PROMPT", "Tell me a short joke.")
THINKER_OUT = "/tmp/omni_thinker_out.json"
WAV_OUT = "/tmp/omni_decoupled.wav"
TIMEOUT_SEC = 600


def run_phase(name, cmd, env, log_path):
    logger.info(f"=== {name} ===  cmd={cmd[:3]}... env_gpu={env.get('CUDA_VISIBLE_DEVICES')}")
    with open(log_path, "w") as logf:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                              timeout=TIMEOUT_SEC, text=True,
                              input=json.dumps({"prompt": PROMPT, "max_new_tokens": 32}))
        dt = time.perf_counter() - t0
    return proc.returncode, dt


def main():
    # Pick GPUs
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpus = [g.strip() for g in visible.split(",") if g.strip()]
    if not gpus:
        logger.error("Set CUDA_VISIBLE_DEVICES to one or two GPU indices")
        return 2
    thinker_gpu = gpus[0]
    talker_gpu = gpus[1] if len(gpus) > 1 else gpus[0]
    logger.info(f"thinker_gpu={thinker_gpu} talker_gpu={talker_gpu}")

    repo = "/root/mateng/rtp-llm"
    py = sys.executable

    # Phase 1: thinker
    if os.path.exists(THINKER_OUT):
        os.unlink(THINKER_OUT)
    thinker_env = os.environ.copy()
    thinker_env["CUDA_VISIBLE_DEVICES"] = thinker_gpu
    thinker_env["PYTHONPATH"] = repo
    thinker_env["OMNI_OUT"] = THINKER_OUT
    thinker_env["OMNI_REQUEST"] = json.dumps({"prompt": PROMPT, "max_new_tokens": 32})
    cmd = [py, "-m", "omni_decoupled.thinker_server"]
    rc, dt = run_phase("THINKER (subprocess)", cmd, thinker_env, "/tmp/thinker_phase.log")
    if rc != 0:
        logger.error(f"thinker phase FAILED rc={rc}; tail of log:")
        with open("/tmp/thinker_phase.log") as f:
            print(f.read()[-3000:])
        return 1
    logger.info(f"thinker phase OK in {dt:.1f}s")

    if not os.path.exists(THINKER_OUT):
        logger.error("thinker did not produce output file")
        return 1
    with open(THINKER_OUT) as f:
        payload = json.load(f)
    logger.info(f"thinker text: {payload['text']!r}")
    logger.info(f"thinker token_ids: {len(payload['token_ids'])}")
    logger.info(f"thinker hidden_states shape: {payload['hidden_shape']}")

    # Phase 2: talker
    if os.path.exists(WAV_OUT):
        os.unlink(WAV_OUT)
    talker_env = os.environ.copy()
    talker_env["CUDA_VISIBLE_DEVICES"] = talker_gpu
    talker_env["PYTHONPATH"] = repo
    talker_env["OMNI_THINKER_OUT"] = THINKER_OUT
    talker_env["OMNI_WAV_OUT"] = WAV_OUT
    cmd = [py, "-m", "omni_decoupled.talker_client"]
    rc, dt = run_phase("TALKER (subprocess)", cmd, talker_env, "/tmp/talker_phase.log")
    if rc != 0:
        logger.error(f"talker phase FAILED rc={rc}; tail of log:")
        with open("/tmp/talker_phase.log") as f:
            print(f.read()[-3000:])
        return 1
    logger.info(f"talker phase OK in {dt:.1f}s")

    if not os.path.exists(WAV_OUT):
        logger.error("talker did not produce wav file")
        return 1
    sz = os.path.getsize(WAV_OUT)
    logger.info(f"WAV: {WAV_OUT} ({sz} bytes)")
    if sz < 1024:
        logger.error("WAV too small — likely empty/corrupt")
        return 1

    logger.info("=" * 70)
    logger.info("DECOUPLED E2E TEST PASSED")
    logger.info(f"  Thinker: {len(payload['token_ids'])} tokens → {payload['text']!r}")
    logger.info(f"  Talker:  WAV {sz} bytes")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
