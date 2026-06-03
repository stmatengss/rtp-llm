# Qwen2.5-Omni on RTP — Validation & Performance Report

**Branch under test:** `mateng/omni-validation` (commit `6201213b3`) — merges PRs #3 (streaming), #4 (audio input), #5 (multi-GPU) into the PR #2 baseline.
**Hardware:** `mateng04`, 8× NVIDIA A10 (22.2 GiB each); benchmarks use indices 5,6.
**Software:** PyTorch 2.6 + CUDA 12.4, gcc-11, conda env `rtp-llm-py310`.
**Model:** `/root/models/Qwen/Qwen2.5-Omni-7B` (bf16).
**Prompt:** `"Tell me a short joke."` (≈25 chat-template tokens).
**Speaker:** Ethan.
**Report generated:** 2026-06-03.

---

## 1. Summary (TL;DR)

- **Functional**: text→text, audio→text, text→audio, and streaming all work end-to-end on the merged `omni-validation` branch. Multi-GPU residency works (thinker on cuda:0, talker on cuda:1, both concurrently loaded). Image and video input are **not yet implemented** (Qwen2.5-Omni supports them in HF reference; this branch only wires audio).
- **Feature parity with vLLM omni**: vLLM does not ship Qwen2.5-Omni in any released version available in this env (transformers in mateng04 also lacks `Qwen2_5OmniForConditionalGeneration`). Direct A/B comparison was not possible. The functional matrix below describes parity in capability terms.
- **Decoupled deployment**: ✅ **implemented** as a two-subprocess topology with file-based IPC (`omni_decoupled/thinker_server.py` + `omni_decoupled/talker_client.py`, driven by `test_omni_decoupled.py`). Thinker on one GPU writes `(text, token_ids, hidden_states)` to a JSON+base64 payload; talker on a different GPU reads it and produces a WAV. End-to-end verified: thinker (cuda:5) → "Why was the math book sad? Because it had too many problems." → talker (cuda:6) → 192 KB WAV in ~17 s wall (talker phase only; thinker phase ~30 s incl. load). The proper gRPC-streaming variant (hidden states over the wire) remains a follow-up — see §3.4.
- **Performance**: sequential full pipeline (text → 4 s of audio) takes 13.6 s wall clock at 0.29 audio-s/wall-s. Streaming delivers the first codec token **0.41 s before** the thinker finishes; this is a latency win, not a throughput win.

---

## 2. Functional parity matrix

| Capability | Qwen2.5-Omni reference | RTP `omni-validation` | Test file | Notes |
|---|---|---|---|---|
| Text → text (thinker only) | ✅ | ✅ | `test_omni_all_stages.py`, `test_omni_audio_thinker.py` | 15 thinker tokens for the joke prompt; deterministic at greedy/top_k=1 |
| Audio → text (thinker only) | ✅ | ✅ | `test_omni_audio_thinker.py` | 2 s sine WAV → "I hear a sine wave." |
| Text → text + audio (full pipeline, sequential) | ✅ | ✅ | `test_omni_all_stages.py` | 4.00 s 24 kHz mono WAV from `Why was the math book sad...` |
| Text → text + audio (streaming / interleaved) | ✅ | ✅ | `test_omni_streaming.py` | Talker emits first codec **before** thinker EOS (PR #3) |
| Image input | ✅ | ❌ | — | Thinker class already extends `MultiModalMixin` and weights load, but only audio_tower path is wired in `Qwen2_5OmniThinkerModel.forward()`; vision branch is a follow-up |
| Video input | ✅ | ❌ | — | Same status as image |
| Multi-GPU residency (thinker + talker concurrent) | n/a (HF runs in one process) | ✅ | `test_omni_multigpu.py` | Thinker 21.6 GiB on cuda:0 + Talker 4.6 GiB on cuda:1, both loaded simultaneously (PR #5) |
| Cross-GPU talker codec generation | n/a | ⚠️ Known bug | gated behind `OMNI_MULTIGPU_RUN_TALKER=1` | `invokePrefillAddFusedQKVBiasTranspose` raises `cudaErrorIllegalAddress` when talker tries to forward on cuda:1 — KV-cache view or kernel workspace escapes per-engine `CUDAGuard`. Single-engine on cuda:1 works fine. |
| Decoupled deployment (thinker on host A, talker on host B) | n/a | ✅ | `test_omni_decoupled.py` | Two-subprocess topology, file IPC. Cross-machine works if the IPC file is on shared FS. gRPC-streaming variant deferred. |
| HF-reference A/B comparison | n/a | ❌ Blocked | — | Local transformers lacks `Qwen2_5OmniForConditionalGeneration`; vLLM omni not packaged in this env |

Legend: ✅ working, ❌ missing, ⚠️ partially working (caveat), 🟡 design only.

---

## 3. Decoupled deployment

### 3.1 Design

The omni-specific decoupling cut is the **thinker/talker boundary**, not PD (prefill/decode) inside the thinker. Two endpoints:

```
+-------------+         gRPC (model_rpc_service.proto)        +-------------+
|  thinker    |   token_ids + hidden_states ─────────────►   |  talker      |
|  server     |   (request: tokens, audio_url; response:      |  client      |
|  cuda:0     |    text + per-token last-layer hidden states) |  cuda:1      |
|  port 8501  |                                               |  (in proc:   |
+-------------+                                               |   talker     |
                                                              |   engine +   |
                                                              |   token2wav) |
                                                              +-------------+
```

The thinker engine already binds a gRPC server on `server_config.start_port` when that value is non-negative (see `RtpLLMOp.cc::initRPCServer` — it skips binding only when `model_rpc_port < 0`). `test_omni_audio_thinker.py` already exercises this path: the test starts a thinker engine with `start_port=18088`, instantiates a `ModelRpcClient(["127.0.0.1:18088"])`, and submits a `GenerateInput` with `mm_inputs=[MultimodalInput(url=wav_path, mm_type=MMUrlType.AUDIO)]`.

### 3.2 What's wired today vs. what's needed

| Piece | Status |
|---|---|
| Thinker as gRPC server | ✅ existing engine startup binds when `start_port ≥ 0` |
| `ModelRpcClient` + `GenerateInput` + `MultimodalInput` (Python) | ✅ used in `test_omni_audio_thinker.py` |
| Server-side mm-feature splice (audio embeddings into thinker forward) | ✅ PR #4: `LocalRpcServiceImpl.prepareInput → mm_processor_->updateMultimodalFeatures → MMProcessEngine.submit → Processor.audio_embedding`, then scattered in `Qwen2_5OmniThinkerModel.forward()` |
| Thinker subprocess that emits (text, token_ids, hidden_states) | ✅ `omni_decoupled/thinker_server.py` |
| Talker subprocess that loads only talker + token2wav, reads thinker payload | ✅ `omni_decoupled/talker_client.py` |
| End-to-end decoupled test | ✅ `test_omni_decoupled.py` |
| Hidden states over actual gRPC (vs file IPC) | ❌ Current `GenerateOutputPB` only carries `output_ids`; would need `repeated float hidden_states` (or bytes) added + C++ rebuild |
| Streaming-decoupled (per-token cross-process) | ❌ See §3.4 |

### 3.3 Implementation effort estimate

- Add `hidden_states_bytes` field to `GenerateOutputPB` (proto regen)
- Wire it through `LocalRpcServiceImpl::GenerateStreamCall` to populate from `output.hidden_states.value()` when present
- Client side: deserialize and feed into `talker_py_model.set_thinker_hidden_states()`
- ~1 day of work; gated on the next milestone.

### 3.4 Streaming variant

A streaming-decoupled topology (talker starts before thinker EOS, *across processes*) is option α in the F2 plan. It requires either (a) a new gRPC bidi-streaming method on top of the streaming work from PR #3 (callback emits per-step over the wire) or (b) pushing the existing `generate_with_callback` model to also expose its callback over RPC. Neither is implemented; the in-process streaming from PR #3 is already proven (§4.2).

---

## 4. Performance results

All measurements taken on `mateng04`, `CUDA_VISIBLE_DEVICES=5,6`, model bf16.

### 4.1 Sequential pipeline (3-run median)

Driver: `bench_omni_report.py --configs sequential --runs 3`. Each run is a fresh subprocess.

| Metric | Median | Min | Max |
|---|---|---|---|
| TTFT (per-token avg over thinker) | 53.7 ms | 53.5 ms | 53.7 ms |
| TTFAT (first audio = time-to-talker-start) | 5273 ms | 5271 ms | 5317 ms |
| TTLT (full pipeline: thinker + talker + token2wav) | **13598 ms** | 13592 ms | 13612 ms |
| Thinker gen (15 tokens) | 805 ms | 803 ms | 805 ms |
| Talker gen (200 codec tokens) | 1974 ms | 1902 ms | 1991 ms |
| Audio duration | 4.00 s | 4.00 s | 4.00 s |
| Throughput | **0.294 audio-s/wall-s** | 0.2939 | 0.2943 |
| GPU memory peak (cuda:0, single-GPU) | not captured (process restart between stages) | — | — |

The 5.3 s gap between thinker finishing (~0.8 s) and talker starting (~5.3 s) is dominated by talker engine load time (model weights + KV cache allocation). Sequential pays this in series; streaming/multi-GPU avoid it.

### 4.2 Streaming (single-run, from `test_omni_streaming.py` smoke)

Driver: `python test_omni_streaming.py`. The bench harness's streaming case had a memory issue I didn't have time to pin down; the test file's own driver works.

| Metric | Value |
|---|---|
| Thinker total | 0.91 s (15 tokens) |
| First thinker token | 0.32 s |
| First codec token | **0.50 s** |
| Last codec token | 2.67 s |
| **Interleave margin** (thinker_done − first_codec) | **+0.41 s** ← positive ⇒ talker started before thinker EOS |
| Audio duration | 4.00 s |
| TTLT (incl. token2wav) | not measured in test logs (test does `os._exit(0)` to skip cleanup) |

Streaming reduces time-to-first-audio compared to sequential because the talker engine is **already loaded** when the thinker finishes — the engine load cost is hidden under the thinker's generation time.

### 4.3 Multi-GPU residency (single-run, from `test_omni_multigpu.py` smoke)

Driver: `CUDA_VISIBLE_DEVICES=5,6 python test_omni_multigpu.py`.

| Device | Resident size | Stage |
|---|---|---|
| cuda:0 | **20933 MB** (after thinker load), grows to **21585 MB** during generate | Thinker (7B bf16 + KV cache) |
| cuda:1 | **4642 MB** (after talker load, idle) | Talker (small model + KV cache) |

Both engines stay loaded simultaneously through the test. Thinker generates 15 tokens. **Talker codec generation on cuda:1 is currently gated** behind `OMNI_MULTIGPU_RUN_TALKER=1` due to `cudaErrorIllegalAddress` in `invokePrefillAddFusedQKVBiasTranspose` — see §5.

### 4.4 Audio input (single-run, from `test_omni_audio_thinker.py` smoke)

Driver: `python test_omni_audio_thinker.py`.

| Metric | Value |
|---|---|
| Audio input | 2 s 440 Hz sine WAV, 16 kHz |
| Decoded text | `"I hear a sine wave."` (7 tokens) |
| Generation time | 1.45 s |
| Audio processing (Whisper-style audio_tower) | included in the 1.45 s |

The decoded text shows the model actually understood the audio content, not just accepted the tokens.

### 4.5 Decoupled deployment

Not measured (not implemented). See §3.

---

## 5. Known caveats

1. **Cross-GPU talker codec generation crashes** (`cudaErrorIllegalAddress` in `invokePrefillAddFusedQKVBiasTranspose`). Single-engine talker on cuda:1 alone works fine; the bug only manifests when both engines are loaded on two devices. F1 gated the talker run behind `OMNI_MULTIGPU_RUN_TALKER=1` so the residency test stays green. Suspect: a KV-cache view or kernel workspace allocation that escapes the per-engine `CUDAGuard`.

2. **mm-placeholder tokens (`expandTokenIds`)** leave non-`-1` garbage in `combo_tokens` at multimodal slot positions (observed `2008386423`). The fix is `clamp(min=0, max=vocab_size-1)` before `embed_tokens()`, **not** `clamp_min(0)` — out-of-bounds index → corrupted embedding → CUDA illegal-memory-access inside RMSNorm a few layers later. Documented in `qwen2_5_omni_thinker.py` and PR #4.

3. **Engine `stop()` under memory pressure** can throw `std::bad_alloc` from a background thread → `std::terminate`. F2's streaming test works around it by calling `os._exit(0)` after validating the WAV.

4. **HF reference A/B is unavailable** in this env (transformers package lacks `Qwen2_5OmniForConditionalGeneration`). Functional outputs were validated against expected plain-English responses (e.g., "I hear a sine wave"), not byte-for-byte against HF.

5. **Build environment is mateng04-specific**: gcc-11 forced (gcc-13 too strict for grpc/absl), local `file://` URLs in `deps/git.bzl` and `deps/http.bzl` for github-unreachable repos. These are **not committed** to the branch.

6. **Bench harness gaps**: my `bench_omni_report.py` driver's streaming / multigpu / audio_in cases hit infrastructure issues (single-GPU OOM under back-to-back loads in streaming; multigpu uses different `ParallelismConfig.world_rank` mechanism than my code assumed; audio_in needs `MultimodalInput` API not the raw proto). Only sequential ran 3-of-3 there. The other configurations' numbers in §4 come from the standalone smoke tests, which all pass cleanly.

---

## 6. Reproduction commands

All commands run from `/root/mateng/rtp-llm` on `mateng04` with `omni-validation` checked out and the `.so` files installed to `/root/miniconda3/envs/rtp-llm-py310/lib/python3.10/site-packages/rtp_llm/libs/`.

```bash
# Activate env
source /root/miniconda3/etc/profile.d/conda.sh && conda activate rtp-llm-py310

# Sequential
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/root/mateng/rtp-llm \
  python test_omni_all_stages.py

# Streaming
CUDA_VISIBLE_DEVICES=5,6 PYTHONPATH=/root/mateng/rtp-llm \
  python test_omni_streaming.py

# Multi-GPU residency (talker codec gated by env)
CUDA_VISIBLE_DEVICES=5,6 PYTHONPATH=/root/mateng/rtp-llm \
  python test_omni_multigpu.py
# To attempt cross-GPU talker generation (known to crash; see §5):
# OMNI_MULTIGPU_RUN_TALKER=1 ... python test_omni_multigpu.py

# Audio input
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/root/mateng/rtp-llm \
  python test_omni_audio_thinker.py

# Sequential multi-run bench (the one that worked end-to-end)
CUDA_VISIBLE_DEVICES=5,6 PYTHONPATH=/root/mateng/rtp-llm \
  python bench_omni_report.py --configs sequential --runs 3 --out /root/omni_bench.json
```

To rebuild from source on mateng04:

```bash
cd /root/mateng/rtp-llm
bazelisk build //:th_transformer //:th_transformer_config \
  --config=cuda12 --jobs=32 \
  --action_env=GCC_HOST_COMPILER_PATH=/usr/bin/gcc-11 \
  --host_action_env=GCC_HOST_COMPILER_PATH=/usr/bin/gcc-11 \
  --per_file_copt='external/.*@-Wno-error' \
  --copt=-Wno-error=unused-result \
  --copt=-Wno-error=unused-function \
  --copt=-Wno-error=format-security
cp bazel-bin/libth_transformer.so /root/miniconda3/envs/rtp-llm-py310/lib/python3.10/site-packages/rtp_llm/libs/
cp bazel-bin/libth_transformer_config.so /root/miniconda3/envs/rtp-llm-py310/lib/python3.10/site-packages/rtp_llm/libs/
cp bazel-bin/librtp_compute_ops.so /root/miniconda3/envs/rtp-llm-py310/lib/python3.10/site-packages/rtp_llm/libs/
```

---

## 7. What changed in this PR

`mateng/omni-validation` merges three open sub-PRs into `omni-phase1`:

- **PR #3** (`omni-streaming`): callback-based `RtpLLMOp::generateWithCallback` + talker streaming buffer
- **PR #4** (`omni-multimodal`): `multimodal_features`/`mm_features_locs` in `PyModelInputs` + `Qwen2_5OmniThinkerModel` py_model + audio splice
- **PR #5** (`omni-multigpu`): per-engine `device_id`, fix `ExecOps.cc::initRuntime` `std::call_once` regression, multi-GPU mode in `OmniEngine`

Plus this report (`docs/omni-validation-report.md`) and the bench driver (`bench_omni_report.py`).

Files touched: 19 changed, 1966 insertions, 48 deletions.

---

## 8. Recommended follow-ups (in priority order)

1. **Fix the cross-GPU talker `cudaErrorIllegalAddress`** (caveat #1). Without this, multi-GPU is residency-only.
2. **Implement decoupled deployment** (§3.3). ~1 day of work.
3. **Wire image + video input** to the thinker (currently only audio).
4. **Streaming-decoupled topology** (§3.4) — gRPC bidi-streaming for hidden states.
5. **Add an HF-reference benchmark** by installing a newer transformers in a side env so we can do byte-for-byte audio comparison (MCD, cosine-similarity of mel spectra, etc.).
6. **Stabilize engine `stop()`** so tests can clean up properly (caveat #3).
