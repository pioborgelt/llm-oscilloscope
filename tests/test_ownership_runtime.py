import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_oscilloscope.ownership import OwnershipCase, build_messages
from llm_oscilloscope.ownership_runtime import (
    OwnershipRuntime,
    PROFILES,
    choice_aliases,
    render_prompt,
    termination_ids,
)


class FakeTokenizer:
    eos_token_id = 2

    def encode(self, text, **kwargs):
        values = {
            "A": [1],
            "a": [3],
            " A": [1],
            " a": [3],
            "B": [4],
            "b": [5],
            " B": [4],
            " b": [5],
        }
        return values[text]

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "<|im_start|>system\nfixed<|im_end|>\n<|im_start|>user\nToday Date: is mentioned in the user's account.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"


class RuntimeHelperTests(unittest.TestCase):
    def test_aliases_are_deduplicated(self):
        self.assertEqual(choice_aliases(FakeTokenizer()), {"A": [1, 3], "B": [4, 5]})

    def test_overlapping_aliases_rejected(self):
        tokenizer = FakeTokenizer()
        tokenizer.encode = lambda *args, **kwargs: [1]
        with self.assertRaises(ValueError):
            choice_aliases(tokenizer)

    def test_eos_union(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2, 7]))
        self.assertEqual(termination_ids(FakeTokenizer(), model), [2, 7])

    def test_render_is_nonmutating_and_uses_native_nonthinking(self):
        case = OwnershipCase("First view", "Second view")
        messages = build_messages(case, "neutral", "ab")
        original = messages[0]["content"]
        tokenizer = FakeTokenizer()
        render_prompt(tokenizer, messages)
        self.assertEqual(messages[0]["content"], original)
        self.assertEqual(
            tokenizer.messages[0]["content"], "/system_override " + original
        )
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_template_drift_is_rejected(self):
        tokenizer = FakeTokenizer()
        tokenizer.apply_chat_template = lambda *args, **kwargs: "assistant: "
        with self.assertRaises(ValueError):
            render_prompt(
                tokenizer,
                build_messages(OwnershipCase("First", "Second"), "user_a", "ba"),
            )

    def test_qwen_profile_does_not_receive_smol_control_word(self):
        tokenizer = FakeTokenizer()
        original_template = tokenizer.apply_chat_template
        tokenizer.apply_chat_template = lambda *a, **kw: (
            original_template(*a, **kw) + "\n"
        )
        messages = build_messages(OwnershipCase("First", "Second"), "user_a", "ba")
        rendered = render_prompt(tokenizer, messages, model="qwen3")
        self.assertEqual(tokenizer.messages[0]["content"], messages[0]["content"])
        self.assertTrue(rendered.endswith(PROFILES["qwen3"]["assistant_suffix"]))
        self.assertNotEqual(
            PROFILES["qwen3"]["aliases"]["B"], PROFILES["smollm3"]["aliases"]["B"]
        )

    def test_unknown_profile_rejected_before_model_import_or_loading(self):
        with self.assertRaises(ValueError):
            OwnershipRuntime(model="unknown-model")

    def test_offline_default_and_opt_in_download_are_explicit(self):
        torch = MagicMock()
        torch.__version__ = "synthetic"
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "synthetic CUDA device"
        transformers = MagicMock()
        transformers.__version__ = "synthetic"
        for allowed in (False, True):
            with (
                patch.dict(
                    "sys.modules", {"torch": torch, "transformers": transformers}
                ),
                patch(
                    "llm_oscilloscope.ownership_runtime.choice_aliases",
                    return_value=PROFILES["qwen3"]["aliases"],
                ),
            ):
                runtime = OwnershipRuntime(
                    model="qwen3", allow_download=allowed, cache_dir="explicit-cache"
                )
            args = transformers.AutoTokenizer.from_pretrained.call_args.kwargs
            self.assertIs(args["local_files_only"], not allowed)
            self.assertIs(args["trust_remote_code"], False)
            self.assertEqual(args["revision"], PROFILES["qwen3"]["revision"])
            self.assertEqual(args["cache_dir"], "explicit-cache")
            self.assertIsNone(runtime.model)
            transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        torch.set_num_threads.assert_not_called()
        torch.manual_seed.assert_not_called()

    def test_all_lengths_checked_before_any_model_evaluation(self):
        runtime = OwnershipRuntime.__new__(OwnershipRuntime)
        runtime._ids = MagicMock(side_effect=[None, None, ValueError("too long")])
        runtime.evaluate = MagicMock()
        runtime.metadata = {"model_id": "synthetic"}
        with self.assertRaises(ValueError):
            runtime.measure(OwnershipCase("First", "Second"))
        runtime.evaluate.assert_not_called()

    def test_close_releases_unused_cuda_allocations(self):
        runtime = OwnershipRuntime.__new__(OwnershipRuntime)
        runtime.model = object()
        runtime.torch = MagicMock()
        runtime.close()
        self.assertIsNone(runtime.model)
        runtime.torch.cuda.empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
