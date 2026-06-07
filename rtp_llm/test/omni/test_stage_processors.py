import unittest

import torch

from rtp_llm.omni.engine.stage_connector import StageOutput


class TestThinker2Talker(unittest.TestCase):
    def test_function_exists(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            thinker2talker,
        )

        self.assertTrue(callable(thinker2talker))

    def test_extracts_embeddings(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            thinker2talker,
        )

        thinker_output = StageOutput(
            token_ids=[1, 2, 3],
            embeddings=torch.randn(1, 10, 3584),
            metadata={"text": "hello"},
        )
        talker_input = thinker2talker(thinker_output)
        self.assertIsNotNone(talker_input.embeddings)
        self.assertEqual(talker_input.embeddings.shape[-1], 3584)
        self.assertEqual(talker_input.metadata["source_text"], "hello")

    def test_preserves_source_token_ids(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            thinker2talker,
        )

        thinker_output = StageOutput(
            token_ids=[10, 20, 30],
            embeddings=torch.randn(1, 5, 3584),
            metadata={},
        )
        talker_input = thinker2talker(thinker_output)
        self.assertEqual(talker_input.metadata["source_token_ids"], [10, 20, 30])


class TestTalker2Code2Wav(unittest.TestCase):
    def test_function_exists(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            talker2code2wav,
        )

        self.assertTrue(callable(talker2code2wav))

    def test_converts_tokens(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            talker2code2wav,
        )

        talker_output = StageOutput(
            token_ids=[10, 20, 30, 40, 50],
            metadata={"codec_tokens": True},
        )
        c2w_input = talker2code2wav(talker_output)
        self.assertIsNotNone(c2w_input.token_ids)
        self.assertEqual(c2w_input.token_ids, [10, 20, 30, 40, 50])
        self.assertTrue(c2w_input.metadata["from_talker"])

    def test_filters_codec_tokens(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            talker2code2wav,
        )

        talker_output = StageOutput(
            token_ids=torch.tensor([100, 200, 8292, 8300, 50]),
            metadata={},
        )
        c2w_input = talker2code2wav(talker_output)
        self.assertEqual(c2w_input.token_ids, [100, 200, 50])


class TestComputeTalkerExternalEmbeddings(unittest.TestCase):
    def test_shape_output(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            compute_talker_external_embeddings,
        )

        seq_len = 5
        embed_dim = 3584
        hidden_size = 896
        vocab_size = 8448

        codec_ids = torch.randint(0, vocab_size, (seq_len,))
        thinker_hs = torch.randn(seq_len, embed_dim)
        embed_w = torch.randn(vocab_size, embed_dim)
        proj_w = torch.randn(hidden_size, embed_dim)
        proj_b = torch.randn(hidden_size)

        result = compute_talker_external_embeddings(
            codec_ids, thinker_hs, embed_w, proj_w, proj_b
        )
        self.assertEqual(result.shape, (seq_len, hidden_size))

    def test_handles_padding(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            compute_talker_external_embeddings,
        )

        seq_len = 10
        hs_len = 5
        embed_dim = 64
        hidden_size = 32

        codec_ids = torch.randint(0, 100, (seq_len,))
        thinker_hs = torch.randn(hs_len, embed_dim)
        embed_w = torch.randn(100, embed_dim)
        proj_w = torch.randn(hidden_size, embed_dim)

        result = compute_talker_external_embeddings(
            codec_ids, thinker_hs, embed_w, proj_w
        )
        self.assertEqual(result.shape, (seq_len, hidden_size))


class TestFuncResolver(unittest.TestCase):
    def test_resolve_valid_function(self):
        from rtp_llm.omni.engine.func_resolver import resolve_func

        func = resolve_func(
            "rtp_llm.omni.models.qwen2_5_omni.stage_processors.thinker2talker"
        )
        self.assertTrue(callable(func))

    def test_resolve_invalid_module(self):
        from rtp_llm.omni.engine.func_resolver import resolve_func

        with self.assertRaises(ModuleNotFoundError):
            resolve_func("nonexistent.module.func")

    def test_resolve_invalid_attribute(self):
        from rtp_llm.omni.engine.func_resolver import resolve_func

        with self.assertRaises(AttributeError):
            resolve_func(
                "rtp_llm.omni.models.qwen2_5_omni.stage_processors.nonexistent_func"
            )

    def test_resolve_no_module_path(self):
        from rtp_llm.omni.engine.func_resolver import resolve_func

        with self.assertRaises(ValueError):
            resolve_func("just_a_name")


if __name__ == "__main__":
    unittest.main()
