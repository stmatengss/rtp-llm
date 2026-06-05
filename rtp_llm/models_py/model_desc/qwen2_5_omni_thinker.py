"""Python model for Qwen2.5-Omni thinker running inside rtp-llm's C++ engine.

The thinker is a standard Qwen3-style decoder-only transformer with one
multimodal twist: at prefill time, audio (and optionally vision) features
replace the placeholder positions in the embedding sequence.

The C++ engine (`MultimodalProcessor::expandTokenIds`) substitutes -1 into
`combo_tokens` at every multimodal-placeholder position before calling forward,
and stages the precomputed feature tensors into
`PyModelInputs.multimodal_features` together with `mm_features_locs` (the
start position of each feature segment). This Python forward:

  1) clamps input_ids to >=0 so embed_tokens() doesn't blow up on -1,
  2) overwrites those positions with the staged mm feature tensors,
  3) runs the standard Qwen3 transformer stack.

For pure-text requests the mm fields are empty and forward() degenerates to
the same code path as Qwen3Model.
"""

from typing import Any, Optional

import torch
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


class Qwen2_5OmniThinkerModel(GptModelBase):
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

    def _splice_mm_features(
        self,
        inputs_embeds: torch.Tensor,
        mm_features,
        mm_features_locs: torch.Tensor,
    ) -> torch.Tensor:
        """Overwrite placeholder rows of inputs_embeds with mm feature tensors.

        mm_features:        list of [N_i, hidden] tensors (CUDA, model dtype)
        mm_features_locs:   int32 tensor of start indices, one per feature
        """
        if mm_features is None:
            return inputs_embeds
        if not isinstance(mm_features, (list, tuple)):
            return inputs_embeds
        if len(mm_features) == 0:
            return inputs_embeds
        if mm_features_locs is None or mm_features_locs.numel() == 0:
            return inputs_embeds

        # mm_features_locs is allocated as pinned int32 on host; pull to CPU list.
        if mm_features_locs.is_cuda:
            locs_list = mm_features_locs.cpu().tolist()
        else:
            locs_list = mm_features_locs.tolist()

        target_dtype = inputs_embeds.dtype
        target_device = inputs_embeds.device
        total = inputs_embeds.shape[0]

        for i, feat in enumerate(mm_features):
            if i >= len(locs_list):
                break
            start = int(locs_list[i])
            n = int(feat.shape[0])
            if n == 0:
                continue
            if start < 0 or start + n > total:
                # Defensive: skip any feature whose splice range is out of bounds.
                continue
            feat_t = feat
            if feat_t.device != target_device:
                feat_t = feat_t.to(target_device)
            if feat_t.dtype != target_dtype:
                feat_t = feat_t.to(target_dtype)
            inputs_embeds[start : start + n] = feat_t

        return inputs_embeds

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids

        # C++ expandTokenIds writes -1 (and may leave other unrelated bytes)
        # at multimodal-placeholder positions in combo_tokens. Clamp to a
        # valid vocab index so embed_tokens() doesn't index out of bounds;
        # the garbage rows produced here are overwritten by the splice below.
        vocab_size = self.embed_tokens.weight.shape[0]
        safe_ids = input_ids.clamp(min=0, max=vocab_size - 1)
        inputs_embeds = self.embed_tokens(safe_ids)

        inputs_embeds = self._splice_mm_features(
            inputs_embeds,
            getattr(inputs, "multimodal_features", None),
            getattr(inputs, "mm_features_locs", None),
        )

        hidden_states = inputs_embeds
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


__all__ = [
    "Qwen2_5OmniThinkerModel",
]
