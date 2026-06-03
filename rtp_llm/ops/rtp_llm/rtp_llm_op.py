import logging
import os
from typing import Dict, List, Optional

from rtp_llm.frontend.token_processor import TokenProcessor
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.ops import RtpLLMOp as CppRtpLLMOp
from rtp_llm.ops import get_block_cache_keys as cpp_get_block_cache_keys
from rtp_llm.utils.mm_process_engine import MMProcessEngine
from rtp_llm.config.engine_config import EngineConfig

class RtpLLMOp:
    def __init__(
        self,
        engine_config: EngineConfig,
        model: BaseModel,
        mm_engine: Optional[MMProcessEngine] = None,
        propose_model: Optional[ProposeModel] = None,
        token_processor: Optional[TokenProcessor] = None,
    ):
        self.engine_config = engine_config
        self.model = model
        self.mm_engine = mm_engine
        self.propose_model = propose_model
        self.ft_op = CppRtpLLMOp()
        self.token_processor = token_processor

    def start(self):
        self.weight = self.model.weight
        logging.info("engine_config: %s", self.engine_config.to_string())
        self.ft_op.init(  # type: ignore
            self.model,
            self.engine_config,
            self.model.vit_config,
            self.mm_engine,
            self.propose_model,
            self.token_processor,
        )

    def stop(self):
        self.ft_op.stop()  # type: ignore

    def generate(self, input_ids, max_new_tokens: int = 4096, eos_token_id: int = -1,
                 return_hidden_states: bool = False):
        """Generate tokens via the C++ engine.

        Args:
            input_ids: 1D int32 tensor of prompt tokens.
            max_new_tokens: cap on tokens to generate.
            eos_token_id: stop word id (-1 to disable).
            return_hidden_states: if True, also return per-token last-layer
                hidden states from the model.

        Returns:
            If return_hidden_states=False (default):
                token_ids: [1, num_generated] int32 tensor (cumulative).
            If return_hidden_states=True:
                (token_ids, hidden_states) tuple, where hidden_states is
                [num_generated, hidden_dim] (one row per generated token).
        """
        if not hasattr(self.ft_op, 'generate'):
            raise RuntimeError(
                "C++ RtpLLMOp.generate() not available. "
                "Rebuild with: bazelisk build //:th_transformer"
            )
        result = self.ft_op.generate(input_ids, max_new_tokens, eos_token_id, return_hidden_states)
        # C++ always returns (token_ids, hidden_states); unwrap when not requested
        # for backwards compatibility.
        if return_hidden_states:
            return result
        token_ids, _ = result
        return token_ids

    def generate_with_callback(self, input_ids, callback,
                               max_new_tokens: int = 4096,
                               eos_token_id: int = -1):
        """Streaming variant of generate(). Invokes `callback` per step.

        The callback signature is:
            callback(token_chunk, hidden_chunk, finished)
        where:
          - token_chunk: int32 CPU tensor of NEW token ids for this step
            (None if no tokens were emitted this step).
          - hidden_chunk: float CPU tensor [n, hidden_dim] of last-layer
            hidden states for the new tokens (None if absent).
          - finished: bool, True on the terminal step.

        Returns the final (token_ids[1, N], hidden_states[N, hidden_dim])
        accumulated across all steps — same shape as
        generate(..., return_hidden_states=True).
        """
        if not hasattr(self.ft_op, 'generate_with_callback'):
            raise RuntimeError(
                "C++ RtpLLMOp.generate_with_callback() not available. "
                "Rebuild with: bazelisk build //:th_transformer"
            )
        return self.ft_op.generate_with_callback(
            input_ids, max_new_tokens, eos_token_id, callback,
        )