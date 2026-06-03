"""Python model for Qwen2.5-Omni talker running inside rtp-llm's C++ engine.

The talker's custom embedding differs from standard QWenV2:
  embed_tokens(codec_token) + thinker_hidden_state → proj(3584→896) → transformer → norm

The C++ engine handles the autoregressive loop, KV cache, and FMHA attention.
This Python model is called by the engine at each step via forward().

Streaming mode
--------------
``push_thinker_hidden_state(hs)`` appends rows to the hidden-state buffer
incrementally (used by F2 interleaved streaming). ``mark_thinker_done()`` is
called once the thinker reaches EOS so ``forward()`` stops blocking when it
runs past the end of the buffer (falling back on the existing pad-with-last-
row behaviour). The non-streaming pipeline still calls
``set_thinker_hidden_states(...)`` followed by ``clear_thinker_hidden_states``
exactly as before — the streaming additions are purely opt-in.
"""

import logging
import threading
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3DecoderLayer
from rtp_llm.models_py.modules import Embedding, RMSNorm
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen2_5OmniTalkerModel(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )

        self.proj_weight = weights.get_global_weight("thinker_to_talker_proj.weight")
        self.proj_bias = weights.get_global_weight("thinker_to_talker_proj.bias")

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

        self._thinker_hidden_states: Optional[torch.Tensor] = None
        self._step: int = 0
        # Streaming support (interleaved thinker→talker, see module docstring).
        self._stream_event = threading.Event()
        self._stream_lock = threading.Lock()
        self._thinker_done: bool = False
        # Hard cap on how long forward() will wait for the next hidden-state
        # chunk before giving up and using the pad-with-last-row behaviour.
        # Keep it generous; the thinker should never take this long per step.
        self._stream_wait_timeout_s: float = 30.0

    def set_thinker_hidden_states(self, hidden_states: torch.Tensor) -> None:
        """Store thinker hidden states before starting generation.

        Args:
            hidden_states: [num_thinker_tokens, embedding_size] tensor
        """
        with self._stream_lock:
            self._thinker_hidden_states = hidden_states.to(
                device=self.proj_weight.device, dtype=self.proj_weight.dtype
            )
            self._step = 0
            # In non-streaming mode the buffer is complete from the start.
            self._thinker_done = True
        self._stream_event.set()

    def clear_thinker_hidden_states(self) -> None:
        with self._stream_lock:
            self._thinker_hidden_states = None
            self._step = 0
            self._thinker_done = False
        self._stream_event.clear()

    def begin_streaming_thinker(self) -> None:
        """Reset state for an interleaved streaming run.

        Call this once before kicking off the thinker.generate_with_callback.
        After this, push_thinker_hidden_state() should be called per step,
        and mark_thinker_done() once when the thinker reaches EOS.
        """
        with self._stream_lock:
            self._thinker_hidden_states = None
            self._step = 0
            self._thinker_done = False
        self._stream_event.clear()

    def push_thinker_hidden_state(self, hidden_states: torch.Tensor) -> None:
        """Append thinker hidden-state rows to the buffer (streaming mode).

        Safe to call from any thread. Notifies any forward() blocked on
        ``self._stream_event``.

        Args:
            hidden_states: [n, embedding_size] tensor (n>=1).
        """
        hs = hidden_states.to(
            device=self.proj_weight.device, dtype=self.proj_weight.dtype
        )
        if hs.dim() == 1:
            hs = hs.unsqueeze(0)
        with self._stream_lock:
            if self._thinker_hidden_states is None:
                self._thinker_hidden_states = hs.contiguous()
            else:
                self._thinker_hidden_states = torch.cat(
                    [self._thinker_hidden_states, hs], dim=0
                ).contiguous()
        self._stream_event.set()

    def mark_thinker_done(self) -> None:
        """Signal the thinker stream has ended; forward() will stop blocking."""
        with self._stream_lock:
            self._thinker_done = True
        self._stream_event.set()

    def _wait_for_hidden_rows(self, needed_end: int) -> None:
        """Block until self._thinker_hidden_states has at least needed_end rows.

        Returns early if the thinker has been marked done. Falls through after
        ``self._stream_wait_timeout_s`` so a stuck producer can't hang the
        talker indefinitely (the existing pad-with-last-row behaviour takes
        over in that case).
        """
        deadline = self._stream_wait_timeout_s
        while True:
            with self._stream_lock:
                have = (
                    self._thinker_hidden_states.shape[0]
                    if self._thinker_hidden_states is not None
                    else 0
                )
                if have >= needed_end or self._thinker_done:
                    return
                self._stream_event.clear()
            # Releases the GIL while waiting (CPython threading semantics).
            got = self._stream_event.wait(timeout=deadline)
            if not got:
                logging.warning(
                    "Qwen2_5OmniTalkerModel: timed out waiting for thinker "
                    "hidden states (have=%d, need>=%d); falling back on padding.",
                    have,
                    needed_end,
                )
                return

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        num_tokens = input_ids.shape[0]

        inputs_embeds = self.embed_tokens(input_ids)

        # In streaming mode, the buffer may not yet have enough rows. Block until
        # either the thinker has produced (self._step + num_tokens) rows or it
        # signals it's done. Non-streaming callers already set _thinker_done=True
        # in set_thinker_hidden_states(), so this returns immediately for them.
        needed_end = self._step + num_tokens
        self._wait_for_hidden_rows(needed_end)

        if self._thinker_hidden_states is not None:
            max_idx = self._thinker_hidden_states.shape[0]
            start = min(self._step, max_idx - 1)
            end = min(self._step + num_tokens, max_idx)
            thinker_hs = self._thinker_hidden_states[start:end]

            if thinker_hs.shape[0] < num_tokens:
                pad_count = num_tokens - thinker_hs.shape[0]
                last_hs = self._thinker_hidden_states[-1:].expand(pad_count, -1)
                thinker_hs = torch.cat([thinker_hs, last_hs], dim=0)

            inputs_embeds = inputs_embeds + thinker_hs
            self._step += num_tokens

        hidden_states = F.linear(inputs_embeds, self.proj_weight, self.proj_bias)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
