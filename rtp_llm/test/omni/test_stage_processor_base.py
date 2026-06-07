import unittest

from rtp_llm.omni.engine.stage_connector import StageOutput
from rtp_llm.omni.engine.func_resolver import resolve_func


def mock_processor(source_output: StageOutput) -> StageOutput:
    new_ids = [x + 100 for x in source_output.token_ids] if source_output.token_ids else None
    return StageOutput(token_ids=new_ids, metadata={"transformed": True})


class TestFuncResolver(unittest.TestCase):
    def test_resolve_builtin_module(self):
        func = resolve_func("os.path.join")
        self.assertTrue(callable(func))

    def test_plain_function_as_processor(self):
        source = StageOutput(token_ids=[1, 2, 3])
        result = mock_processor(source)
        self.assertEqual(result.token_ids, [101, 102, 103])
        self.assertTrue(result.metadata["transformed"])


if __name__ == "__main__":
    unittest.main()
