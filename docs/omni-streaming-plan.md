# F2 — Interleaved streaming thinker→talker

## Goal
Talker starts producing codec tokens as soon as the thinker emits its first
hidden state, instead of waiting for the thinker to reach EOS. Reduces
time-to-first-audio. Matches HF reference behavior.

## API choice: B (callback)
- Add `RtpLLMOp::generateWithCallback(input_ids, max_new_tokens, eos_token_id, callback)`.
- `callback` is a Python callable invoked from the streaming loop after every
  step with `(new_token_ids, hidden_states)` (both CPU tensors).
- We acquire the GIL just for the callback dispatch and release it again
  before pulling the next chunk. Avoids the pybind11 generator dance.
- Returns the final `(token_ids, hidden_states)` like `generate()` so existing
  callers can keep working.

## Concurrency model
- Two engines resident at the same time (thinker on one device, talker on
  another). Each `RtpLLMOp` instance owns its own engine.
- A dedicated talker Python thread calls
  `talker_engine.generate(initial=[BOS], max_new_tokens=200, eos=EOS)`.
- The talker's `forward()` reads `self._thinker_hidden_states[step]`; if the
  requested row is not yet available it blocks on a `threading.Event` until
  the thinker callback pushes it (or the thinker signals "done").
- The thinker thread calls `generateWithCallback`, the callback runs
  `talker_py_model.push_thinker_hidden_state(hs)` which appends the row and
  notifies the event.
- When the thinker emits EOS, the test calls
  `talker_py_model.mark_thinker_done()` so future short reads just fall back
  on the last row (the existing pad-with-last-row behaviour).

## Files touched
- `rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h` — declare new method.
- `rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc` — implement + bind it.
- `rtp_llm/ops/rtp_llm/rtp_llm_op.py` — Python wrapper passthrough.
- `rtp_llm/models_py/model_desc/qwen2_5_omni_talker.py`:
  - `push_thinker_hidden_state(hs)` appends + notifies.
  - `mark_thinker_done()` flips a flag so `forward()` stops waiting.
  - `forward()` waits on the event when `_step` outruns the buffer
    (unless `_thinker_done` is set).
- `test_omni_streaming.py` — interleaved e2e test with latency measurements.

## Definition of done
- `test_omni_streaming.py` passes.
- Reports `t_first_codec < t_thinker_done` (proves interleaving).
- Final WAV non-empty, no NaNs.
- `test_omni_all_stages.py` still passes (sequential path untouched).
