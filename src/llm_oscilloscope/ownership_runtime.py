"""Pinned, opt-in local SmolLM3 and Qwen3 profiles for the ownership assay."""

from __future__ import annotations

from contextlib import redirect_stdout
import inspect
from pathlib import Path
import sys

from .ownership import (
    ARMS,
    ORDERS,
    ChoiceReading,
    OwnershipCase,
    build_messages,
    run_assay,
)


MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
MODEL_REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"
EXPECTED_ALIASES = {"A": [32, 64, 264, 362], "B": [33, 65, 293, 426]}
PROFILES = {
    "smollm3": {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "aliases": EXPECTED_ALIASES,
        "system_override": True,
        "assistant_suffix": "<|im_start|>assistant\n<think>\n\n</think>\n",
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "aliases": {"A": [32, 64, 264, 362], "B": [33, 65, 293, 425]},
        "system_override": False,
        "assistant_suffix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    },
}


def choice_aliases(tokenizer) -> dict[str, list[int]]:
    aliases = {}
    for letter in ("A", "B"):
        values = set()
        for text in (letter, letter.lower(), " " + letter, " " + letter.lower()):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                values.add(int(encoded[0]))
        aliases[letter] = sorted(values)
    if not all(aliases.values()) or set(aliases["A"]) & set(aliases["B"]):
        raise ValueError("A and B need disjoint, nonempty single-token aliases")
    return aliases


def termination_ids(tokenizer, model) -> list[int]:
    values = set()
    for candidate in (tokenizer.eos_token_id, model.generation_config.eos_token_id):
        if candidate is not None:
            values.update(
                candidate if isinstance(candidate, (list, tuple, set)) else [candidate]
            )
    if not values:
        raise ValueError("The model profile needs an EOS token")
    return sorted(int(value) for value in values)


def render_prompt(
    tokenizer, messages: list[dict[str, str]], *, model: str = "smollm3"
) -> str:
    if model not in PROFILES:
        raise ValueError("Unknown ownership model profile")
    profile = PROFILES[model]
    copied = [dict(message) for message in messages]
    if len(copied) != 2 or [r["role"] for r in copied] != ["system", "user"]:
        raise ValueError("The assay requires its fixed system/user prompt pair")
    if profile["system_override"]:
        copied[0]["content"] = "/system_override " + copied[0]["content"]
    rendered = tokenizer.apply_chat_template(
        copied,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    if not rendered.endswith(profile["assistant_suffix"]):
        raise ValueError(
            "Unexpected non-thinking template; do not silently change the measurement point"
        )
    header = rendered.split("<|im_start|>user\n", 1)[0]
    if "Today Date:" in header:
        raise ValueError("Unexpected dated system metadata in the pinned assay profile")
    return rendered


class OwnershipRuntime:
    """Batch-one FP16/SDPA profile. No network unless allow_download is explicit."""

    def __init__(
        self,
        *,
        model: str = "smollm3",
        cache_dir: str | Path | None = None,
        allow_download: bool = False,
    ):
        if model not in PROFILES:
            raise ValueError("Unknown ownership model profile")
        self.profile_name = model
        profile = PROFILES[model]
        self.model_id = profile["model_id"]
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Live assays need the optional dependencies: pip install 'llm-oscilloscope[sycophancy]'"
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The live assay profile requires CUDA. Recorded assay replay is GPU-free."
            )
        self.torch = torch
        self.model = None
        self.model_class = AutoModelForCausalLM
        self.load_args = {
            "revision": profile["revision"],
            "local_files_only": not allow_download,
            "trust_remote_code": False,
        }
        if cache_dir is not None:
            self.load_args["cache_dir"] = str(cache_dir)
        with redirect_stdout(sys.stderr):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, **self.load_args
            )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.aliases = choice_aliases(self.tokenizer)
        if self.aliases != profile["aliases"]:
            raise ValueError(
                "Tokenizer aliases differ from the pinned ownership profile"
            )
        self.metadata = {
            "model_id": self.model_id,
            "model_revision": profile["revision"],
            "profile": "fp16_sdpa_batch1_native_nonthinking"
            + ("_system_override" if profile["system_override"] else ""),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "float16",
            "quantization": None,
            "batch_size": 1,
            "attention": "sdpa",
            "max_new_tokens": 8,
            "max_prompt_tokens": 1536,
            "choice_alias_ids": self.aliases,
            "measurement_position": "last prompt position before the answer, after native empty non-thinking prefix",
        }

    def _ids(self, messages):
        text = render_prompt(self.tokenizer, messages, model=self.profile_name)
        ids = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        if ids.shape[1] > 1536:
            raise ValueError(
                "Assay prompt exceeds 1536 tokens; shorten both accounts. No text was truncated."
            )
        return ids

    def _load_model(self) -> None:
        if self.model is not None:
            return
        # No global thread-count or sampling-state changes in the calling application.
        with redirect_stdout(sys.stderr):
            self.model = self.model_class.from_pretrained(
                self.model_id,
                **self.load_args,
                dtype=self.torch.float16,
                attn_implementation="sdpa",
                device_map={"": "cuda:0"},
            )
        self.model.eval()
        self.eos_ids = termination_ids(self.tokenizer, self.model)
        parameters = inspect.signature(self.model.forward).parameters
        self.last_logits = {}
        if "logits_to_keep" in parameters:
            self.last_logits["logits_to_keep"] = 1
        elif "num_logits_to_keep" in parameters:
            self.last_logits["num_logits_to_keep"] = 1

    def evaluate(self, messages) -> ChoiceReading:
        ids = self._ids(messages)
        self._load_model()
        torch = self.torch
        ids = ids.to(self.model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                repetition_penalty=1.0,
                max_new_tokens=8,
                return_dict_in_generate=True,
                output_logits=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_ids,
                **self.last_logits,
            )
            logits = output.logits[0][0].float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Model returned nonfinite logits")
            combined = torch.stack(
                [
                    torch.logsumexp(logits[self.aliases[letter]], dim=0)
                    for letter in ("A", "B")
                ]
            )
            mass = float(
                torch.exp(
                    torch.logsumexp(combined, dim=0) - torch.logsumexp(logits, dim=0)
                )
            )
            generated = output.sequences[0, ids.shape[1] :].tolist()
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            ended = bool(generated and generated[-1] in self.eos_ids)
            return ChoiceReading(
                float(combined[0]), float(combined[1]), mass, text, ended
            )

    def measure(self, case: OwnershipCase) -> dict:
        # Reject every overlength condition before allocating model weights or evaluating any arm.
        for arm in ARMS:
            for order in ORDERS:
                self._ids(build_messages(case, arm, order))
        return run_assay(case, self.evaluate, runtime=self.metadata)

    def close(self) -> None:
        self.model = None
        # Return unused allocations to other GPU applications when the assay ends.
        self.torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


class SmolOwnershipRuntime(OwnershipRuntime):
    """Explicit reference-profile convenience class."""

    def __init__(
        self, *, cache_dir: str | Path | None = None, allow_download: bool = False
    ):
        super().__init__(
            model="smollm3", cache_dir=cache_dir, allow_download=allow_download
        )
