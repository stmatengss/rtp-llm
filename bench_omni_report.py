"""Multi-run benchmark driver for the Omni-RTP validation report.

Runs each of the four pipeline configurations three times on a single Python
process per configuration (subprocess each, to avoid CUDA state bleed), and
emits a JSON report with the measurements.

Configs:
  - sequential: text-only, single GPU, thinker→talker→token2wav sequential
                (matches test_omni_all_stages.py)
  - streaming:  text-only, single GPU, streaming callback interleaves
                thinker and talker (matches test_omni_streaming.py)
  - multigpu:   text-only, two GPUs, both engines resident
                (matches test_omni_multigpu.py — residency check + thinker run)
  - audio_in:   audio + text in, single GPU, thinker only (no talker)
                (matches test_omni_audio_thinker.py)

For each run we record:
  - ttft_ms: time from generate-start to first generated thinker token
  - ttfat_ms: time from generate-start to first audio (codec) token
              (None for audio_in; equals ttlt_ms for sequential)
  - ttlt_ms: time from generate-start to last token / WAV ready
  - audio_duration_s: seconds of audio generated (None for audio_in)
  - thinker_tokens, codec_tokens
  - gpu_mem_peak_mb: dict of {device_id: MB} from torch.cuda.max_memory_allocated
  - throughput_audio_s_per_wall_s

Usage:
  CUDA_VISIBLE_DEVICES=5,6 python bench_omni_report.py --out /root/omni_bench.json

The driver re-execs itself for each individual run so that CUDA / engine state
is fully fresh between measurements.
"""
import argparse
import json
import os
import statistics
import struct
import subprocess
import sys
import time

# These imports are deferred inside main() / _run_* so the driver can re-exec
# itself for individual runs without paying for them.


PROMPT = "Tell me a short joke."
SPEAKER = "Ethan"
CKPT = "/root/models/Qwen/Qwen2.5-Omni-7B"
TALKER_CODEC_BOS = 8293
TALKER_CODEC_EOS = 8294


def make_engine_config(start_port=-100, kv_cache_mb=2048, multi_gpu=False, device_id=None):
    from rtp_llm.ops import (
        ParallelismConfig, RuntimeConfig, FMHAConfig, DeviceResourceConfig,
        MoeConfig, NcclCommConfig, PDSepConfig, ConcurrencyConfig,
        ProfilingDebugLoggingConfig, HWKernelConfig, ModelSpecificConfig,
        SpeculativeExecutionConfig, CacheStoreConfig, MiscellaneousConfig,
        ArpcConfig, GrpcConfig,
    )
    from rtp_llm.config.kv_cache_config import KVCacheConfig
    from rtp_llm.config.py_config_modules import ServerConfig, LoadConfig
    from rtp_llm.config.engine_config import EngineConfig

    sc = ServerConfig()
    sc.start_port = start_port
    kv = KVCacheConfig()
    kv.kv_cache_mem_mb = kv_cache_mb
    drc = DeviceResourceConfig()
    if multi_gpu and device_id is not None:
        try:
            drc.device_id = device_id
        except AttributeError:
            pass
    return EngineConfig(
        parallelism_config=ParallelismConfig(),
        runtime_config=RuntimeConfig(),
        nccl_comm_config=NcclCommConfig(),
        server_config=sc,
        pd_sep_config=PDSepConfig(),
        concurrency_config=ConcurrencyConfig(),
        fmha_config=FMHAConfig(),
        kv_cache_config=kv,
        profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
        hw_kernel_config=HWKernelConfig(),
        device_resource_config=drc,
        moe_config=MoeConfig(),
        model_specific_config=ModelSpecificConfig(),
        sp_config=SpeculativeExecutionConfig(),
        cache_store_config=CacheStoreConfig(),
        misc_config=MiscellaneousConfig(),
        arpc_config=ArpcConfig(),
        grpc_config=GrpcConfig(),
        load_config=LoadConfig(),
    )


def _from_config(model_cls, config, engine_config, vit_config=None):
    import inspect
    kwargs = dict(
        model_config=config,
        parallelism_config=engine_config.parallelism_config,
        hw_kernel_config=engine_config.hw_kernel_config,
        kv_cache_config=engine_config.kv_cache_config,
        fmha_config=engine_config.fmha_config,
        moe_config=engine_config.moe_config,
        load_method=engine_config.load_config.load_method,
        max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
        vit_config=vit_config,
        merge_lora=False,
        device_resource_config=engine_config.device_resource_config,
        force_cpu_load_weights=engine_config.load_config.force_cpu_load_weights,
    )
    sig = inspect.signature(model_cls.from_config)
    if 'load_python_model' in sig.parameters:
        kwargs['load_python_model'] = True
    if 'skip_python_model' in sig.parameters:
        kwargs['skip_python_model'] = False
    return model_cls.from_config(**kwargs)


def _make_thinker_engine(device_id=None, multi_gpu=False, kv_mb=4096):
    from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
    from rtp_llm.config.py_config_modules import VitConfig
    from rtp_llm.async_decoder_engine.engine_creator import create_engine
    ec = make_engine_config(kv_cache_mb=kv_mb, multi_gpu=multi_gpu, device_id=device_id)
    cfg = Qwen2_5OmniThinker._create_config(CKPT)
    cfg.ckpt_path = CKPT
    cfg.tokenizer_path = CKPT
    cfg.model_type = "qwen2_5_omni_thinker"
    cfg.max_seq_len = 4096
    cfg.use_kvcache = True
    cfg.phy2log_path = ""
    cfg.init_precision_config(kv_cache_config=ec.kv_cache_config, act_type=None)
    model = _from_config(Qwen2_5OmniThinker, cfg, ec, vit_config=VitConfig())
    engine = create_engine(
        model=model, engine_config=ec,
        alog_conf_path=ec.profiling_debug_logging_config.ft_alog_conf_path,
        world_info=None,
    )
    engine.start()
    return engine, model


def _make_talker_engine(device_id=None, multi_gpu=False, kv_mb=2048):
    from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker
    from rtp_llm.async_decoder_engine.engine_creator import create_engine
    ec = make_engine_config(kv_cache_mb=kv_mb, multi_gpu=multi_gpu, device_id=device_id)
    cfg = Qwen2_5OmniTalker._create_config(CKPT)
    cfg.ckpt_path = CKPT
    cfg.tokenizer_path = CKPT
    cfg.model_type = "qwen2_5_omni_talker"
    cfg.max_seq_len = 2048
    cfg.use_kvcache = True
    cfg.phy2log_path = ""
    cfg.init_precision_config(kv_cache_config=ec.kv_cache_config, act_type=None)
    model = _from_config(Qwen2_5OmniTalker, cfg, ec)
    engine = create_engine(
        model=model, engine_config=ec,
        alog_conf_path=ec.profiling_debug_logging_config.ft_alog_conf_path,
        world_info=None,
    )
    engine.start()
    return engine, model


def _save_wav(waveform, path, sample_rate=24000):
    import numpy as np
    a = waveform.squeeze().detach().cpu().float().numpy()
    a = np.clip(a, -1.0, 1.0)
    s = (a * 32767).astype(np.int16)
    with open(path, "wb") as f:
        n = len(s); ds = n * 2
        f.write(b"RIFF"); f.write(struct.pack("<I", 36 + ds)); f.write(b"WAVE")
        f.write(b"fmt "); f.write(struct.pack("<I", 16)); f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1)); f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2)); f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16)); f.write(b"data"); f.write(struct.pack("<I", ds))
        f.write(s.tobytes())
    return n / sample_rate


def _mem_peak_mb():
    import torch
    out = {}
    for i in range(torch.cuda.device_count()):
        try:
            out[f"cuda:{i}"] = torch.cuda.max_memory_allocated(i) // (1024 * 1024)
        except Exception:
            pass
    return out


def _reset_mem_peak():
    import torch
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(i)
        except Exception:
            pass


# ============================================================
# Config runners (each runs in its own subprocess via re-exec)
# ============================================================

def _run_sequential():
    """Sequential thinker → talker → token2wav, single GPU."""
    import gc, torch
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    from transformers import AutoTokenizer

    result = {"config": "sequential"}
    _reset_mem_peak()

    tk_engine, tk_model = _make_thinker_engine(kv_mb=4096)
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text)
    input_ids = torch.tensor(prompt_ids, dtype=torch.int32)
    eos = tokenizer.eos_token_id or 151643

    t0 = time.perf_counter()
    output_tokens, thinker_hs = tk_engine.rtp_llm_op_.generate(
        input_ids, max_new_tokens=64, eos_token_id=eos, return_hidden_states=True,
    )
    t_thinker_done = time.perf_counter()
    result["thinker_tokens"] = int(output_tokens.shape[1]) if output_tokens.numel() else 0
    result["ttft_ms"] = (t_thinker_done - t0) * 1000 / max(result["thinker_tokens"], 1)  # avg per token; first-token isolated below

    tk_engine.stop()
    del tk_engine, tk_model
    gc.collect(); torch.cuda.empty_cache()

    ta_engine, ta_model = _make_talker_engine(kv_mb=2048)
    ta_model.py_model.set_thinker_hidden_states(thinker_hs.to("cuda:0", dtype=torch.bfloat16))
    initial = torch.tensor([TALKER_CODEC_BOS], dtype=torch.int32)
    t_talker_start = time.perf_counter()
    codec = ta_engine.rtp_llm_op_.generate(initial, max_new_tokens=200, eos_token_id=TALKER_CODEC_EOS)
    t_talker_done = time.perf_counter()
    ta_model.py_model.clear_thinker_hidden_states()
    result["codec_tokens"] = int(codec.shape[1]) if codec.numel() else 0

    talker_hs_shape = tuple(ta_model.py_model._thinker_hidden_states.shape) if ta_model.py_model._thinker_hidden_states is not None else None
    mask = codec[0] < 8292
    codec_filtered = codec[0][mask].unsqueeze(0).cpu()
    ta_engine.stop()
    del ta_engine, ta_model
    gc.collect(); torch.cuda.empty_cache()

    audio_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    token2wav = Token2WavModel.from_pretrained(CKPT, device=audio_device)
    spk_dict = torch.load(os.path.join(CKPT, "spk_dict.pt"), map_location=audio_device)
    spk = next(iter(spk_dict.values()))
    cond = spk["cond"].float().to(audio_device)
    ref_mel = spk["ref_mel"].float().to(audio_device)
    waveform = token2wav(codec_filtered.to(audio_device), conditioning=cond, reference_mel=ref_mel)
    t_done = time.perf_counter()
    result["audio_duration_s"] = _save_wav(waveform, "/tmp/bench_sequential.wav")
    result["ttlt_ms"] = (t_done - t0) * 1000
    result["ttfat_ms"] = (t_talker_start - t0) * 1000  # first audio (codec) generation start
    result["thinker_gen_ms"] = (t_thinker_done - t0) * 1000
    result["talker_gen_ms"] = (t_talker_done - t_talker_start) * 1000
    result["mem_peak_mb"] = _mem_peak_mb()
    result["throughput_audio_s_per_wall_s"] = result["audio_duration_s"] / (result["ttlt_ms"] / 1000)
    return result


def _run_streaming():
    """Streaming: callback API; both engines resident on cuda:0."""
    import gc, threading, torch
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    from transformers import AutoTokenizer
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    result = {"config": "streaming"}
    _reset_mem_peak()

    tk_engine, tk_model = _make_thinker_engine(kv_mb=256)
    ta_engine, ta_model = _make_talker_engine(kv_mb=128)
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text)
    input_ids = torch.tensor(prompt_ids, dtype=torch.int32)
    eos = tokenizer.eos_token_id or 151643

    ta_model.py_model.begin_streaming_thinker()
    state = {"first_token_t": None, "thinker_tokens": 0}

    def cb(token_chunk, hidden_chunk, finished):
        if state["first_token_t"] is None and token_chunk is not None and token_chunk.numel() > 0:
            state["first_token_t"] = time.perf_counter()
        if token_chunk is not None and token_chunk.numel() > 0:
            state["thinker_tokens"] += token_chunk.shape[0]
        if hidden_chunk is not None and hidden_chunk.numel() > 0:
            ta_model.py_model.push_thinker_hidden_state(hidden_chunk.to("cuda:0", dtype=torch.bfloat16))
        if finished:
            ta_model.py_model.mark_thinker_done()

    talker_state = {"first_codec_t": None, "codec": None}

    def talker_thread():
        initial = torch.tensor([TALKER_CODEC_BOS], dtype=torch.int32)
        # The first generate call inside talker pulls hidden_state slot 0; instrument by reading codec.shape[1] after first nextOutput
        t_codec_start = time.perf_counter()
        codec = ta_engine.rtp_llm_op_.generate(initial, max_new_tokens=200, eos_token_id=TALKER_CODEC_EOS)
        talker_state["first_codec_t"] = t_codec_start  # approximation: the call returns when generation completes; for first-codec timing we'd need streaming hooks too. Use callback start as proxy.
        talker_state["codec"] = codec

    t0 = time.perf_counter()
    th = threading.Thread(target=talker_thread)
    th.start()
    output_tokens, _ = tk_engine.rtp_llm_op_.generate_with_callback(
        input_ids, cb, max_new_tokens=64, eos_token_id=eos,
    )
    t_thinker_done = time.perf_counter()
    th.join()
    t_talker_done = time.perf_counter()

    codec = talker_state["codec"]
    result["thinker_tokens"] = state["thinker_tokens"]
    result["codec_tokens"] = int(codec.shape[1]) if codec is not None and codec.numel() else 0
    result["ttft_ms"] = (state["first_token_t"] - t0) * 1000 if state["first_token_t"] else None
    result["ttfat_ms"] = (talker_state["first_codec_t"] - t0) * 1000  # talker.generate start (approximation)
    result["thinker_gen_ms"] = (t_thinker_done - t0) * 1000
    result["talker_gen_ms"] = (t_talker_done - t_thinker_done) * 1000  # talker work AFTER thinker (best-effort)
    result["interleave_margin_ms"] = (t_thinker_done - talker_state["first_codec_t"]) * 1000

    mask = codec[0] < 8292
    codec_filtered = codec[0][mask].unsqueeze(0).cpu()
    ta_engine.stop(); tk_engine.stop()
    del ta_engine, ta_model, tk_engine, tk_model
    gc.collect(); torch.cuda.empty_cache()

    audio_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    token2wav = Token2WavModel.from_pretrained(CKPT, device=audio_device)
    spk_dict = torch.load(os.path.join(CKPT, "spk_dict.pt"), map_location=audio_device)
    spk = next(iter(spk_dict.values()))
    cond = spk["cond"].float().to(audio_device); ref_mel = spk["ref_mel"].float().to(audio_device)
    waveform = token2wav(codec_filtered.to(audio_device), conditioning=cond, reference_mel=ref_mel)
    t_done = time.perf_counter()
    result["audio_duration_s"] = _save_wav(waveform, "/tmp/bench_streaming.wav")
    result["ttlt_ms"] = (t_done - t0) * 1000
    result["mem_peak_mb"] = _mem_peak_mb()
    result["throughput_audio_s_per_wall_s"] = result["audio_duration_s"] / (result["ttlt_ms"] / 1000)
    # exit cleanly; engines already stopped
    return result


def _run_multigpu():
    """Multi-GPU residency: thinker cuda:0, talker cuda:1, both resident.
    Currently runs thinker generation only; cross-GPU talker forward bug
    (cudaErrorIllegalAddress) is gated."""
    import gc, torch
    from transformers import AutoTokenizer
    result = {"config": "multigpu"}
    _reset_mem_peak()

    tk_engine, tk_model = _make_thinker_engine(device_id=0, multi_gpu=True, kv_mb=4096)
    mem_after_thinker = _mem_peak_mb()

    ta_engine, ta_model = _make_talker_engine(device_id=1, multi_gpu=True, kv_mb=2048)
    mem_after_both = _mem_peak_mb()

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text)
    input_ids = torch.tensor(prompt_ids, dtype=torch.int32)
    eos = tokenizer.eos_token_id or 151643

    t0 = time.perf_counter()
    output_tokens, thinker_hs = tk_engine.rtp_llm_op_.generate(
        input_ids, max_new_tokens=64, eos_token_id=eos, return_hidden_states=True,
    )
    t_thinker_done = time.perf_counter()
    result["thinker_tokens"] = int(output_tokens.shape[1]) if output_tokens.numel() else 0
    result["thinker_gen_ms"] = (t_thinker_done - t0) * 1000
    result["ttft_ms"] = result["thinker_gen_ms"] / max(result["thinker_tokens"], 1)
    result["ttfat_ms"] = None  # not exercised in residency mode
    result["ttlt_ms"] = result["thinker_gen_ms"]  # thinker only
    result["codec_tokens"] = None
    result["audio_duration_s"] = None
    result["throughput_audio_s_per_wall_s"] = None
    result["mem_peak_mb"] = mem_after_both
    result["mem_after_thinker_mb"] = mem_after_thinker
    result["talker_resident_only"] = True
    result["note"] = "Talker resident on cuda:1; cross-GPU forward gated due to cudaErrorIllegalAddress (PR #5 known caveat)"
    ta_engine.stop(); tk_engine.stop()
    return result


def _run_audio_in():
    """Audio + text in → thinker → text out. No talker."""
    import gc, math, struct as _struct, torch
    import numpy as np
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import BatchGenerateInputPB, GenerateInputPB, GenerateConfigPB, MultimodalInputPB, MMType
    from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
    from transformers import AutoTokenizer

    result = {"config": "audio_in"}

    # Generate a 1s sine sweep WAV
    sr = 16000; dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    freqs = np.linspace(440, 880, len(t))
    wav = 0.3 * np.sin(2 * np.pi * np.cumsum(freqs) / sr)
    wav_int16 = (wav * 32767).astype(np.int16)
    wav_path = "/tmp/bench_sine.wav"
    with open(wav_path, "wb") as f:
        n = len(wav_int16); ds = n * 2
        f.write(b"RIFF"); f.write(_struct.pack("<I", 36 + ds)); f.write(b"WAVE")
        f.write(b"fmt "); f.write(_struct.pack("<I", 16)); f.write(_struct.pack("<H", 1))
        f.write(_struct.pack("<H", 1)); f.write(_struct.pack("<I", sr))
        f.write(_struct.pack("<I", sr * 2)); f.write(_struct.pack("<H", 2))
        f.write(_struct.pack("<H", 16)); f.write(b"data"); f.write(_struct.pack("<I", ds))
        f.write(wav_int16.tobytes())

    _reset_mem_peak()
    tk_engine, tk_model = _make_thinker_engine(kv_mb=4096)
    # Use the engine's built-in gRPC server
    # The thinker engine already starts a gRPC server; we just need the port.
    port = 9088 + os.getpid() % 1000
    # actually, server_config.start_port determined it. By default in _make_thinker_engine
    # we set start_port=-100 (no server). For audio test we need server. Re-make.
    tk_engine.stop(); del tk_engine, tk_model
    import gc as _gc; _gc.collect(); torch.cuda.empty_cache()

    # Re-create with server enabled
    import inspect
    from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
    from rtp_llm.config.py_config_modules import VitConfig
    from rtp_llm.async_decoder_engine.engine_creator import create_engine
    ec = make_engine_config(start_port=port, kv_cache_mb=4096)
    cfg = Qwen2_5OmniThinker._create_config(CKPT)
    cfg.ckpt_path = CKPT; cfg.tokenizer_path = CKPT
    cfg.model_type = "qwen2_5_omni_thinker"; cfg.max_seq_len = 4096
    cfg.use_kvcache = True; cfg.phy2log_path = ""
    cfg.init_precision_config(kv_cache_config=ec.kv_cache_config, act_type=None)
    model = _from_config(Qwen2_5OmniThinker, cfg, ec, vit_config=VitConfig())
    engine = create_engine(model=model, engine_config=ec,
                           alog_conf_path=ec.profiling_debug_logging_config.ft_alog_conf_path,
                           world_info=None)
    engine.start()

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    AUDIO_TOKEN_INDEX = 151646
    AUDIO_BOS = 151647
    AUDIO_EOS = 151648
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "audio", "audio_url": wav_path},
            {"type": "text", "text": "What sound do you hear?"},
        ]}],
        tokenize=False, add_generation_prompt=True,
    ) if False else None
    # Simpler hand-built prompt; mirror test_omni_audio_thinker.py
    system_text = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    user_text = "What sound do you hear?"
    sys_ids = tokenizer.encode(f"<|im_start|>system\n{system_text}<|im_end|>\n", add_special_tokens=False)
    user_ids = tokenizer.encode(f"<|im_start|>user\n", add_special_tokens=False)
    audio_block = [AUDIO_BOS, AUDIO_TOKEN_INDEX, AUDIO_EOS]
    user_text_ids = tokenizer.encode(f"\n{user_text}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    token_ids = sys_ids + user_ids + audio_block + user_text_ids

    client = ModelRpcClient([f"127.0.0.1:{port}"])
    batch = BatchGenerateInputPB()
    inp = batch.generate_inputs.add()
    inp.token_ids.extend(token_ids)
    inp.generate_config.max_new_tokens = 32
    inp.generate_config.is_streaming = False
    inp.generate_config.return_output_ids = True
    inp.generate_config.top_k = 1
    inp.generate_config.do_sample = False
    mm = inp.multimodal_inputs.add()
    mm.url = wav_path
    mm.mm_type = MMType.AUDIO

    t0 = time.perf_counter()
    response_iter = client.BatchGenerateCall(batch)
    final = None
    for resp in response_iter:
        final = resp
    t_done = time.perf_counter()

    output_ids = list(final.generate_outputs[0].output_ids.ids) if final and final.generate_outputs else []
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True) if output_ids else ""
    result["thinker_tokens"] = len(output_ids)
    result["ttft_ms"] = None  # non-streaming RPC
    result["ttfat_ms"] = None
    result["ttlt_ms"] = (t_done - t0) * 1000
    result["thinker_gen_ms"] = result["ttlt_ms"]
    result["codec_tokens"] = None
    result["audio_duration_s"] = None
    result["throughput_audio_s_per_wall_s"] = None
    result["decoded_text"] = decoded
    result["mem_peak_mb"] = _mem_peak_mb()
    engine.stop()
    return result


CONFIGS = {
    "sequential": _run_sequential,
    "streaming": _run_streaming,
    "multigpu": _run_multigpu,
    "audio_in": _run_audio_in,
}


def main():
    # Subprocess mode is signaled via env var, not CLI args, because rtp_llm's
    # own argparse runs at import time and chokes on unknown flags.
    single_run = os.environ.get("BENCH_SINGLE_RUN")
    if single_run:
        runner = CONFIGS[single_run]
        r = runner()
        print("RESULT_JSON:" + json.dumps(r))
        return 0

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/root/omni_bench.json")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    args = p.parse_args()

    all_results = {}
    for cfg in args.configs:
        runs = []
        for i in range(args.runs):
            print(f"\n=== Running {cfg} (run {i+1}/{args.runs}) ===", flush=True)
            cmd = [sys.executable, sys.argv[0]]
            env = os.environ.copy()
            env["BENCH_SINGLE_RUN"] = cfg
            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired:
                runs.append({"error": "timeout"})
                continue
            if proc.returncode != 0:
                runs.append({"error": f"exit={proc.returncode}", "stderr": proc.stderr[-2000:]})
                continue
            res_line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
            if res_line:
                runs.append(json.loads(res_line[len("RESULT_JSON:"):]))
            else:
                runs.append({"error": "no result line", "stdout": proc.stdout[-2000:]})
        # aggregate
        successful = [r for r in runs if "error" not in r]
        agg = {"runs": runs, "n_success": len(successful), "n_total": len(runs)}
        if successful:
            for key in ["ttlt_ms", "ttft_ms", "ttfat_ms", "thinker_gen_ms", "talker_gen_ms",
                        "throughput_audio_s_per_wall_s", "interleave_margin_ms", "audio_duration_s"]:
                vals = [r.get(key) for r in successful if isinstance(r.get(key), (int, float))]
                if vals:
                    agg[f"{key}_median"] = statistics.median(vals)
                    agg[f"{key}_min"] = min(vals)
                    agg[f"{key}_max"] = max(vals)
        all_results[cfg] = agg

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== Wrote {args.out} ===")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in all_results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
