import torch
import torch.nn.functional as F
from typing import Optional

from rtp_llm.omni.engine.stage_connector import StageOutput


def thinker2talker(source_output: StageOutput) -> StageOutput:
    """Transform thinker output for talker input.

    Passes thinker hidden states as embeddings, which the talker will use
    to compute external_embeddings = embed(codec) + thinker_hs -> proj.
    """
    return StageOutput(
        embeddings=source_output.embeddings,
        metadata={
            "source_token_ids": source_output.token_ids,
            "source_text": source_output.metadata.get("text", ""),
        },
    )


def talker2code2wav(source_output: StageOutput) -> StageOutput:
    """Transform talker output for token2wav input.

    Filters codec tokens (< 8292) and passes them to the vocoder.
    """
    token_ids = source_output.token_ids
    if token_ids is not None and isinstance(token_ids, torch.Tensor):
        mask = token_ids < 8292
        token_ids = token_ids[mask].tolist()
    return StageOutput(
        token_ids=token_ids,
        metadata={"from_talker": True},
    )


def compute_talker_external_embeddings(
    codec_token_ids: torch.Tensor,
    thinker_hidden_states: torch.Tensor,
    embed_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute pre-projected embeddings for talker C++ engine.

    The talker's custom embedding: embed(codec) + thinker_hs -> linear_proj.
    Result is passed as input_embeddings to the C++ engine which skips
    its own embed_tokens() step.

    Args:
        codec_token_ids: [seq_len] codec token IDs
        thinker_hidden_states: [seq_len, embed_dim] thinker hidden states
        embed_weight: [vocab_size, embed_dim] talker embedding table
        proj_weight: [hidden_size, embed_dim] projection weight
        proj_bias: [hidden_size] optional projection bias

    Returns:
        [seq_len, hidden_size] projected embeddings ready for decoder layers
    """
    embeds = F.embedding(codec_token_ids, embed_weight)
    seq_len = codec_token_ids.shape[0]
    hs_len = thinker_hidden_states.shape[0]
    if seq_len > hs_len:
        padding = torch.zeros(
            seq_len - hs_len,
            thinker_hidden_states.shape[1],
            device=thinker_hidden_states.device,
            dtype=thinker_hidden_states.dtype,
        )
        thinker_hs = torch.cat([thinker_hidden_states, padding], dim=0)
    else:
        thinker_hs = thinker_hidden_states[:seq_len]
    combined = embeds + thinker_hs
    return F.linear(combined, proj_weight, proj_bias)
