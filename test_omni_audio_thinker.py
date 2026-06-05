"""End-to-end test: audio + text → thinker → text response via the RTP C++ engine.

Validates the multimodal input path:
  Stage 0:  Audio file (WAV) is preprocessed by Processor.audio_embedding via the
            already-wired MMProcessEngine, returning audio features.
  Stage 1:  Thinker (LanguageCppEngine) receives a chat prompt that contains the
            <|AUDIO|> placeholder token (id 151646) and a list of mm_inputs with
            the WAV path. The C++ LocalRpcServer.prepareInput auto-invokes
            mm_processor_->updateMultimodalFeatures(input), which calls back into
            Python's MMProcessEngine.submit() to compute features, then expands
            input_ids in place. The engine then runs the thinker with audio +
            text context and generates a text response.

This test exercises the path with NO C++ rebuild — the multimodal wiring is
already present in the engine; we just hadn't been driving it from a test.

The low-level path (rtp_llm_op_.generate(input_ids,...)) used by
test_omni_all_stages.py only handles text. Here we go through the gRPC server
that the engine starts on init() (when start_port > 0), submitting via
ModelRpcClient.

Usage:
    CUDA_VISIBLE_DEVICES=4 python test_omni_audio_thinker.py
"""
import asyncio
import inspect
import logging
import os
import struct
import sys
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_omni_audio_thinker")

CKPT = "/root/models/Qwen/Qwen2.5-Omni-7B"
WAV_PATH = "/root/test_omni_thinker_input.wav"
PROMPT_TEXT = "What do you hear in this audio? Answer in one short sentence."


def create_test_wav(path, duration=2.0, freq=440, sample_rate=16000):
    """Write a sine-tone WAV to `path`. Mirrors test_audio_processor.py."""
    if os.path.exists(path):
        return path
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    audio_int16 = (audio * 32767).astype(np.int16)
    with open(path, "wb") as f:
        num_samples = len(audio_int16)
        data_size = num_samples * 2
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int16.tobytes())
    logger.info(f"Created test WAV: {path} ({duration}s, {sample_rate}Hz, {freq}Hz tone)")
    return path


def create_engine_config(start_port=8088, kv_cache_mb=4096):
    """Engine config with a real start_port so the gRPC server actually binds."""
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

    server_config = ServerConfig()
    server_config.start_port = start_port

    kv_cache_config = KVCacheConfig()
    kv_cache_config.kv_cache_mem_mb = kv_cache_mb
    kv_cache_config.test_block_num = 0

    return EngineConfig(
        parallelism_config=ParallelismConfig(),
        runtime_config=RuntimeConfig(),
        nccl_comm_config=NcclCommConfig(),
        server_config=server_config,
        pd_sep_config=PDSepConfig(),
        concurrency_config=ConcurrencyConfig(),
        fmha_config=FMHAConfig(),
        kv_cache_config=kv_cache_config,
        profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
        hw_kernel_config=HWKernelConfig(),
        device_resource_config=DeviceResourceConfig(),
        moe_config=MoeConfig(),
        model_specific_config=ModelSpecificConfig(),
        sp_config=SpeculativeExecutionConfig(),
        cache_store_config=CacheStoreConfig(),
        misc_config=MiscellaneousConfig(),
        arpc_config=ArpcConfig(),
        grpc_config=GrpcConfig(),
        load_config=LoadConfig(),
    )


def from_config_with_python_model(model_cls, config, engine_config, vit_config=None):
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


def make_engine(model_cls, ckpt_path, engine_config, model_type, max_seq_len=4096, vit_config=None):
    from rtp_llm.async_decoder_engine.engine_creator import create_engine

    config = model_cls._create_config(ckpt_path)
    config.ckpt_path = ckpt_path
    config.tokenizer_path = ckpt_path
    config.model_type = model_type
    config.max_seq_len = max_seq_len
    config.use_kvcache = True
    config.phy2log_path = ""
    config.init_precision_config(
        kv_cache_config=engine_config.kv_cache_config, act_type=None
    )

    model = from_config_with_python_model(model_cls, config, engine_config, vit_config)
    engine = create_engine(
        model=model,
        engine_config=engine_config,
        alog_conf_path=engine_config.profiling_debug_logging_config.ft_alog_conf_path,
        world_info=None,
    )
    engine.start()
    return engine, model


def build_audio_chat_prompt(tokenizer, wav_path, text):
    """Render a Qwen2.5-Omni chat template with one audio + one text turn.

    Returns: (prompt_str, input_ids_list)
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": wav_path},
            {"type": "text", "text": text},
        ]},
    ]
    # Use the model's bundled chat_template.json which emits
    # <|audio_bos|><|AUDIO|><|audio_eos|> for audio content.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt)
    return prompt, input_ids


async def submit_audio_request(engine, model_rpc_port, prompt_input_ids, wav_path,
                                eos_token_id, max_new_tokens=64):
    """Submit a multimodal request to the engine via its gRPC port."""
    from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
    from rtp_llm.utils.base_model_datatypes import GenerateConfig, GenerateInput
    from rtp_llm.utils.multimodal_util import MMUrlType, MultimodalInput

    client = ModelRpcClient(
        addresses=[f"127.0.0.1:{model_rpc_port}"],
        client_config={},
    )

    generate_config = GenerateConfig()
    generate_config.max_new_tokens = max_new_tokens
    generate_config.do_sample = False
    generate_config.top_k = 1
    generate_config.is_streaming = False
    generate_config.return_output_ids = True
    if eos_token_id is not None and eos_token_id >= 0:
        generate_config.stop_words_list = [[int(eos_token_id)]]

    mm_inputs = [MultimodalInput(url=wav_path, mm_type=MMUrlType.AUDIO)]
    input_ids_t = torch.tensor(prompt_input_ids, dtype=torch.int32)

    gen_input = GenerateInput(
        request_id=int(time.time() * 1000) % 100_000_000,
        token_ids=input_ids_t,
        mm_inputs=mm_inputs,
        generate_config=generate_config,
        tokenizer=None,
    )

    final_outputs = None
    async for outputs in client.enqueue(gen_input):
        final_outputs = outputs
    return final_outputs


def main():
    from rtp_llm.config.py_config_modules import VitConfig
    from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
    from transformers import AutoTokenizer

    logger.info("=" * 70)
    logger.info("Audio-input thinker test (Qwen2.5-Omni, RTP C++ engine)")
    logger.info("=" * 70)

    create_test_wav(WAV_PATH, duration=2.0, freq=440, sample_rate=16000)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    eos_id = tokenizer.eos_token_id or 151645
    audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    logger.info(f"<|AUDIO|> token id = {audio_token_id} (expected 151646)")

    prompt_str, prompt_ids = build_audio_chat_prompt(tokenizer, WAV_PATH, PROMPT_TEXT)
    logger.info(f"Prompt (first 200 chars): {prompt_str[:200]!r}")
    logger.info(f"Tokenized to {len(prompt_ids)} ids; "
                f"<|AUDIO|> appears at indices: "
                f"{[i for i, t in enumerate(prompt_ids) if t == audio_token_id]}")
    assert audio_token_id in prompt_ids, (
        "Chat template did not emit <|AUDIO|> placeholder — check template/version"
    )

    # Build engine with a real start_port so gRPC server binds.
    start_port = int(os.environ.get("OMNI_TEST_START_PORT", "18088"))
    logger.info(f"\n=== Building thinker engine (start_port={start_port}) ===")
    vit_config = VitConfig()
    engine_config = create_engine_config(start_port=start_port, kv_cache_mb=4096)
    rpc_port = engine_config.server_config.rpc_server_port
    logger.info(f"Engine gRPC will listen on 127.0.0.1:{rpc_port}")

    t0 = time.time()
    engine, model = make_engine(
        Qwen2_5OmniThinker, CKPT, engine_config, "qwen2_5_omni_thinker",
        max_seq_len=4096, vit_config=vit_config,
    )
    logger.info(f"Thinker engine started in {time.time()-t0:.1f}s")

    # Sanity: the engine must have an mm_engine wired up.
    assert engine.mm_engine is not None, (
        "LanguageCppEngine.mm_engine is None — Qwen2_5OmniThinker should be multimodal"
    )
    logger.info(f"engine.mm_engine = {engine.mm_engine}; "
                f"model.mm_part = {type(model.mm_part).__name__}")

    # Brief settle; the gRPC server thread initializes asynchronously.
    time.sleep(2.0)

    try:
        logger.info("\n=== Submitting audio + text to thinker via gRPC ===")
        t0 = time.time()
        outputs = asyncio.run(submit_audio_request(
            engine, rpc_port, prompt_ids, WAV_PATH, eos_id, max_new_tokens=64,
        ))
        elapsed = time.time() - t0
        logger.info(f"Generation took {elapsed:.2f}s")

        assert outputs is not None and outputs.generate_outputs, (
            "No outputs returned from gRPC stream"
        )
        out = outputs.generate_outputs[0]
        token_ids = out.output_ids
        if token_ids is None or token_ids.numel() == 0:
            logger.error("Engine returned an empty output_ids tensor")
            return 1

        gen_ids = token_ids.flatten().tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        logger.info(f"Thinker generated {len(gen_ids)} tokens")
        logger.info(f"Decoded: {gen_text!r}")

        # Loose validations: not empty, not just an EOS, no NaN.
        assert len(gen_text.strip()) > 0, "Decoded text is empty"
        assert all(0 <= t < 200_000 for t in gen_ids), "Suspicious token ids"

    finally:
        logger.info("Stopping engine...")
        engine.stop()

    logger.info("\n" + "=" * 70)
    logger.info("AUDIO-INPUT THINKER TEST PASSED")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
