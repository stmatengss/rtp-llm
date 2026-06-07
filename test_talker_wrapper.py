"""Test TalkerEngineWrapper weight access and generation logic."""
import logging
import torch
import json
import os

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_talker_wrapper")

CKPT = "/root/models/Qwen/Qwen2.5-Omni-7B"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def test_talker_weight_loading():
    """Test that we can load talker weights and access them correctly."""
    from safetensors.torch import load_file

    logger.info("=== Testing talker weight loading ===")

    index_path = os.path.join(CKPT, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Find all talker weights
    talker_prefix = "talker."
    talker_shards = set()
    talker_keys = []
    for key in index["weight_map"]:
        if key.startswith(talker_prefix):
            talker_shards.add(index["weight_map"][key])
            talker_keys.append(key)

    logger.info(f"Found {len(talker_keys)} talker weights in {len(talker_shards)} shards")

    # Load a subset to check shapes
    for shard in list(talker_shards)[:1]:
        shard_path = os.path.join(CKPT, shard)
        weights = load_file(shard_path)
        for k, v in weights.items():
            if k.startswith(talker_prefix):
                short_k = k[len(talker_prefix):]
                if any(x in short_k for x in ["embed_tokens", "codec_head", "thinker_to_talker_proj"]):
                    logger.info(f"  {short_k}: {v.shape} {v.dtype}")

    # Check specific weight shapes
    all_weights = {}
    for shard in talker_shards:
        shard_path = os.path.join(CKPT, shard)
        w = load_file(shard_path)
        for k, v in w.items():
            if k.startswith(talker_prefix):
                all_weights[k[len(talker_prefix):]] = v

    embed = all_weights["model.embed_tokens.weight"]
    codec_head = all_weights["codec_head.weight"]
    proj_w = all_weights["thinker_to_talker_proj.weight"]
    proj_b = all_weights["thinker_to_talker_proj.bias"]

    logger.info(f"embed_tokens: {embed.shape}")
    logger.info(f"codec_head: {codec_head.shape}")
    logger.info(f"thinker_to_talker_proj.weight: {proj_w.shape}")
    logger.info(f"thinker_to_talker_proj.bias: {proj_b.shape}")

    assert embed.shape == (8448, 3584), f"Unexpected embed shape: {embed.shape}"
    assert proj_w.shape == (896, 3584), f"Unexpected proj_w shape: {proj_w.shape}"
    assert proj_b.shape == (896,), f"Unexpected proj_b shape: {proj_b.shape}"

    logger.info("Weight loading PASSED")


def test_embed_and_project():
    """Test the embedding + projection logic directly."""
    from safetensors.torch import load_file

    logger.info("=== Testing embed + project ===")

    index_path = os.path.join(CKPT, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    talker_prefix = "talker."
    all_weights = {}
    shards = set()
    for key in index["weight_map"]:
        if key.startswith(talker_prefix):
            shards.add(index["weight_map"][key])
    for shard in shards:
        w = load_file(os.path.join(CKPT, shard))
        for k, v in w.items():
            if k.startswith(talker_prefix):
                all_weights[k[len(talker_prefix):]] = v.to(DEVICE).to(DTYPE)

    embed_w = all_weights["model.embed_tokens.weight"]
    proj_w = all_weights["thinker_to_talker_proj.weight"]
    proj_b = all_weights["thinker_to_talker_proj.bias"]

    # Test embedding + projection
    token_ids = torch.tensor([0, 1, 2, 100], dtype=torch.long, device=DEVICE)
    thinker_hs = torch.randn(4, 3584, dtype=DTYPE, device=DEVICE)

    embeds = torch.nn.functional.embedding(token_ids, embed_w)
    assert embeds.shape == (4, 3584), f"Bad embed shape: {embeds.shape}"

    combined = embeds + thinker_hs
    projected = torch.nn.functional.linear(combined, proj_w, proj_b)
    assert projected.shape == (4, 896), f"Bad proj shape: {projected.shape}"

    assert not projected.isnan().any(), "NaN in projected"
    logger.info(f"embed: {embeds.shape} -> combined: {combined.shape} -> proj: {projected.shape}")
    logger.info("Embed + project PASSED")


if __name__ == "__main__":
    test_talker_weight_loading()
    test_embed_and_project()
    logger.info("\nAll tests passed!")
