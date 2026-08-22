import unittest

from llm_oscilloscope.qwen_runtime import (
    _llm_int8_hardware_issue,
    _termination_ids,
)


class _GenerationConfig:
    eos_token_id = [2, 3]


class _Model:
    generation_config = _GenerationConfig()


class _Tokenizer:
    eos_token_id = 1


class QwenRuntimeHelperTests(unittest.TestCase):
    def test_termination_ids_merge_model_and_tokenizer(self):
        self.assertEqual(_termination_ids(_Tokenizer(), _Model()), {1, 2, 3})

    def test_llm_int8_hardware_gate_accepts_turing_or_newer(self):
        self.assertIsNone(_llm_int8_hardware_issue(True, (7, 5)))
        self.assertIsNone(_llm_int8_hardware_issue(True, (8, 6)))

    def test_llm_int8_hardware_gate_explains_failures(self):
        self.assertIn("CUDA", _llm_int8_hardware_issue(False, None))
        self.assertIn("below", _llm_int8_hardware_issue(True, (6, 1)))


if __name__ == "__main__":
    unittest.main()
