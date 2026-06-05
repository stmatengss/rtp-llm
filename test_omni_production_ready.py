"""Production-ready end-to-end test for Qwen2.5-Omni on RTP.

Single test driver that runs each subsystem as an isolated subprocess and
reports a final per-stage pass/fail + overall verdict. Designed to be the
gate test for "this branch is production-ready for omni serving."

Each stage runs in its own Python subprocess to avoid CUDA state bleed
and to keep failures isolated. Stages:

  1. text_only          — sequential thinker → talker → token2wav (single GPU)
  2. streaming          — interleaved thinker/talker callback (single GPU)
  3. audio_input        — sine WAV → thinker → text (single GPU)
  4. multigpu_residency — thinker cuda:0 + talker cuda:1 both resident (2 GPUs)
  5. decoupled          — thinker subprocess → file IPC → talker subprocess (2 GPUs)

Pass criteria per stage are documented inline. The driver exits 0 if ALL
stages pass; non-zero with a summary table otherwise.

Usage:
  CUDA_VISIBLE_DEVICES=5,6 python test_omni_production_ready.py
  # or to run a subset:
  OMNI_PROD_STAGES=text_only,streaming python test_omni_production_ready.py
"""
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO,
                    format="prod-e2e %(levelname)s %(message)s")
logger = logging.getLogger("test_omni_production_ready")

REPO = os.environ.get("OMNI_REPO_ROOT", "/root/mateng/rtp-llm")
PY = sys.executable
TIMEOUT_PER_STAGE = int(os.environ.get("OMNI_PROD_TIMEOUT", "900"))

# Each entry: (stage_id, description, command, env_overrides, pass_check)
# pass_check is a callable(returncode, log_str) -> (bool, brief_reason)

def _check_text_only(rc, log):
    if rc != 0:
        return False, f"exit={rc}"
    if "E2E ALL-STAGES TEST PASSED" not in log:
        return False, "no PASSED marker"
    if "Generated text: " not in log:
        return False, "no generated text"
    if "WAV via token2wav" not in log:
        return False, "no WAV generated"
    return True, "passed"


def _check_streaming(rc, log):
    if rc != 0:
        return False, f"exit={rc}"
    if "F2 STREAMING TEST PASSED" not in log:
        return False, "no PASSED marker"
    if "Interleaving margin" not in log:
        return False, "no interleave margin reported"
    # Find the margin value and check it's positive (talker started before thinker done)
    for line in log.splitlines():
        if "Interleaving margin:" in line:
            try:
                # format: "Interleaving margin: t_thinker_done - t_first_codec = 0.907s - 0.496s = 0.411s"
                margin_s = float(line.rsplit("=", 1)[1].strip().rstrip("s"))
                if margin_s <= 0:
                    return False, f"interleave margin not positive ({margin_s:.3f}s)"
                return True, f"interleave margin {margin_s:.3f}s"
            except (ValueError, IndexError):
                pass
    return True, "passed"


def _check_audio_input(rc, log):
    if rc != 0:
        return False, f"exit={rc}"
    if "AUDIO-INPUT THINKER TEST PASSED" not in log:
        return False, "no PASSED marker"
    if "Decoded: " not in log:
        return False, "no decoded text"
    return True, "passed"


def _check_multigpu(rc, log):
    if rc != 0:
        return False, f"exit={rc}"
    if "MULTI-GPU TP RESIDENCY TEST PASSED" not in log:
        return False, "no PASSED marker"
    # Check both devices show resident memory
    if "cuda:0" not in log or "cuda:1" not in log:
        return False, "missing cuda:0 or cuda:1 residency report"
    return True, "passed"


def _check_decoupled(rc, log):
    if rc != 0:
        return False, f"exit={rc}"
    if "DECOUPLED E2E TEST PASSED" not in log:
        return False, "no PASSED marker"
    if "WAV: " not in log:
        return False, "no WAV produced"
    return True, "passed"


STAGES = [
    {
        "id": "text_only",
        "desc": "sequential text→audio (2 GPUs: engines on cuda:0, token2wav on cuda:1)",
        "cmd": [PY, os.path.join(REPO, "test_omni_all_stages.py")],
        # Needs 2 GPUs because token2wav routes to cuda:1 to dodge engine's
        # leaked CUDA buffers on cuda:0 (see report §5 caveat).
        "env": {"CUDA_VISIBLE_DEVICES_OVERRIDE": "first2"},
        "check": _check_text_only,
    },
    {
        "id": "streaming",
        "desc": "interleaved streaming (single GPU)",
        "cmd": [PY, os.path.join(REPO, "test_omni_streaming.py")],
        "env": {"CUDA_VISIBLE_DEVICES_OVERRIDE": "first2"},
        "check": _check_streaming,
    },
    {
        "id": "audio_input",
        "desc": "audio input → thinker → text (single GPU)",
        "cmd": [PY, os.path.join(REPO, "test_omni_audio_thinker.py")],
        "env": {"CUDA_VISIBLE_DEVICES_OVERRIDE": "first1"},
        "check": _check_audio_input,
    },
    {
        "id": "multigpu_residency",
        "desc": "thinker cuda:0 + talker cuda:1 residency (2 GPUs)",
        "cmd": [PY, os.path.join(REPO, "test_omni_multigpu.py")],
        "env": {"CUDA_VISIBLE_DEVICES_OVERRIDE": "first2"},
        "check": _check_multigpu,
    },
    {
        "id": "decoupled",
        "desc": "thinker subprocess + talker subprocess (file IPC, 2 GPUs)",
        "cmd": [PY, os.path.join(REPO, "test_omni_decoupled.py")],
        "env": {"CUDA_VISIBLE_DEVICES_OVERRIDE": "first2"},
        "check": _check_decoupled,
    },
]


def select_gpus(visible_str, mode):
    gpus = [g.strip() for g in visible_str.split(",") if g.strip()]
    if mode == "first1":
        return ",".join(gpus[:1]) if gpus else visible_str
    if mode == "first2":
        return ",".join(gpus[:2]) if len(gpus) >= 2 else visible_str
    return visible_str


def run_stage(stage, base_visible):
    log_path = f"/tmp/prod_e2e_{stage['id']}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO
    if "CUDA_VISIBLE_DEVICES_OVERRIDE" in stage["env"]:
        env["CUDA_VISIBLE_DEVICES"] = select_gpus(base_visible, stage["env"]["CUDA_VISIBLE_DEVICES_OVERRIDE"])
    for k, v in stage["env"].items():
        if k != "CUDA_VISIBLE_DEVICES_OVERRIDE":
            env[k] = v

    t0 = time.perf_counter()
    try:
        with open(log_path, "w") as f:
            proc = subprocess.run(stage["cmd"], env=env, stdout=f,
                                  stderr=subprocess.STDOUT, timeout=TIMEOUT_PER_STAGE)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        return False, f"timeout after {TIMEOUT_PER_STAGE}s", time.perf_counter() - t0, log_path
    dt = time.perf_counter() - t0

    with open(log_path) as f:
        log = f.read()
    ok, reason = stage["check"](rc, log)
    return ok, reason, dt, log_path


def main():
    base_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not base_visible:
        logger.error("Set CUDA_VISIBLE_DEVICES (need 2 GPUs for full suite)")
        return 2

    requested = os.environ.get("OMNI_PROD_STAGES", "")
    if requested:
        wanted = set(s.strip() for s in requested.split(","))
        stages = [s for s in STAGES if s["id"] in wanted]
    else:
        stages = STAGES

    logger.info(f"running {len(stages)} stages on GPUs: {base_visible}")
    logger.info(f"per-stage timeout: {TIMEOUT_PER_STAGE}s")
    logger.info("")

    results = []
    for i, stage in enumerate(stages, 1):
        logger.info(f"[{i}/{len(stages)}] {stage['id']}: {stage['desc']}")
        ok, reason, dt, log_path = run_stage(stage, base_visible)
        marker = "PASS" if ok else "FAIL"
        logger.info(f"   {marker} ({reason}) in {dt:.1f}s  log={log_path}")
        results.append((stage["id"], ok, reason, dt, log_path))

    logger.info("")
    logger.info("=" * 70)
    logger.info("PRODUCTION-READY E2E SUMMARY")
    logger.info("=" * 70)
    n_pass = sum(1 for _, ok, *_ in results if ok)
    for sid, ok, reason, dt, log_path in results:
        marker = "✅" if ok else "❌"
        logger.info(f"  {marker} {sid:25s} {dt:6.1f}s  {reason}")
    logger.info("=" * 70)
    logger.info(f"OVERALL: {n_pass}/{len(results)} stages passed")
    logger.info("=" * 70)

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
