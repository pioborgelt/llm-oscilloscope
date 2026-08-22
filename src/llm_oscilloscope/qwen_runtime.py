"""Transparent token-by-token Qwen runtime for the CLI research preview."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .channels import QwenResearchChannels, load_cli_config


MIN_LLM_INT8_CAPABILITY = (7, 5)


@dataclass(frozen=True)
class TokenReading:
    position: int
    token_id: int
    text: str
    token_probability: float
    channels: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "token_id": self.token_id,
            "text": self.text,
            "token_probability": self.token_probability,
            **self.channels,
        }


class SelectedLayerCapture:
    """Capture only the final sequence position from selected decoder layers."""

    def __init__(self, model: Any, layers: tuple[int, ...]):
        self.states: dict[int, np.ndarray] = {}
        self.handles = [
            model.model.layers[layer].register_forward_hook(self._hook(layer))
            for layer in layers
        ]

    def _hook(self, layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            # Copy only the scored position; a full prefill can be much larger.
            self.states[layer] = (
                value[:, -1, :].detach().float().cpu().numpy()[0].astype(np.float32)
            )

        return hook

    def reset(self) -> None:
        self.states = {}

    def take(self, layer: int) -> np.ndarray:
        try:
            return self.states[layer]
        except KeyError as error:
            raise RuntimeError(f"Qwen layer {layer} was not captured") from error

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _termination_ids(tokenizer: Any, model: Any) -> set[int]:
    values: set[int] = set()
    for candidate in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple, set)):
            values.update(int(value) for value in candidate)
        else:
            values.add(int(candidate))
    return values


def _sample_token(torch: Any, logits: Any, temperature: float, generator: Any) -> Any:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).reshape(-1)


def _llm_int8_hardware_issue(
    cuda_available: bool, capability: tuple[int, int] | None
) -> str | None:
    """Return a clear incompatibility reason for the bitsandbytes LLM.int8 path."""

    if not cuda_available:
        return "CUDA is not available"
    if capability is None:
        return "CUDA compute capability could not be read"
    if capability < MIN_LLM_INT8_CAPABILITY:
        required = ".".join(str(value) for value in MIN_LLM_INT8_CAPABILITY)
        observed = ".".join(str(value) for value in capability)
        return (
            f"GPU compute capability {observed} is below the LLM.int8 "
            f"minimum {required}"
        )
    return None


class QwenInstrumentedRuntime:
    """Pinned Qwen runtime sharing one incremental path for generate and replay."""

    def __init__(
        self,
        *,
        artifacts_root: str | Path | None = None,
        quantization: str = "8bit",
        device_map: str = "auto",
        max_gpu_memory: str = "6200MiB",
        max_cpu_memory: str = "48GiB",
        allow_download: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Live generation needs the optional runtime dependencies. "
                "Install with: pip install -e '.[runtime]'"
            ) from error

        if quantization not in {"8bit", "none"}:
            raise ValueError("quantization must be '8bit' or 'none'")
        config = load_cli_config()
        model_id = config["model_id"]
        revision = config["model_revision"]
        token = os.environ.get("HF_TOKEN")
        local_only = not allow_download

        load_mode = "downloads allowed" if allow_download else "local cache only"
        print(f"Loading model ({load_mode})", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            token=token,
            local_files_only=local_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_arguments: dict[str, Any] = {
            "revision": revision,
            "token": token,
            "local_files_only": local_only,
            "device_map": device_map,
        }
        if quantization == "8bit":
            cuda_available = bool(torch.cuda.is_available())
            capability = (
                tuple(int(value) for value in torch.cuda.get_device_capability(0))
                if cuda_available
                else None
            )
            hardware_issue = _llm_int8_hardware_issue(cuda_available, capability)
            if hardware_issue is not None:
                raise RuntimeError(
                    f"The evidence-matched 8-bit profile is unavailable: {hardware_issue}. "
                    "Use --quantization none only if sufficient memory is available, and "
                    "treat the resulting scores as runtime-shifted."
                )
            try:
                importlib.import_module("bitsandbytes")
            except (ImportError, OSError, RuntimeError) as error:
                raise RuntimeError(
                    "8-bit loading needs a working bitsandbytes installation. "
                    "Reinstall with: python -m pip install -e \".[runtime]\""
                ) from error
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as error:
                raise RuntimeError("8-bit loading needs bitsandbytes") from error
            model_arguments.update(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                    ),
                    "max_memory": {0: max_gpu_memory, "cpu": max_cpu_memory},
                    "dtype": torch.float16,
                }
            )
        else:
            model_arguments["dtype"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_arguments)
        model.eval()
        layers = config["layers"]
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.channels = QwenResearchChannels(artifacts_root)
        self.detector_layer = int(layers["detector"])
        self.subject_layer = int(layers["subject_routing"])
        self.capture = SelectedLayerCapture(
            model, (self.detector_layer, self.subject_layer)
        )
        self.quantization = quantization

    def close(self) -> None:
        self.capture.close()

    def __enter__(self) -> "QwenInstrumentedRuntime":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    def _prompt_ids(self, prompt: str) -> Any:
        values = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return values.to(self.model.get_input_embeddings().weight.device)

    def _step_stream(
        self,
        prompt: str,
        *,
        forced_ids: list[int] | None,
        max_new_tokens: int,
        temperature: float,
        seed: int,
    ) -> Iterator[TokenReading]:
        torch = self.torch
        prompt_ids = self._prompt_ids(prompt)
        stops = _termination_ids(self.tokenizer, self.model)
        generated_ids: list[int] = []
        previous_text = ""
        generator = torch.Generator(device=prompt_ids.device)
        generator.manual_seed(seed)

        with torch.inference_mode():
            self.capture.reset()
            output = self.model(input_ids=prompt_ids, use_cache=True, return_dict=True)
            past = output.past_key_values
            logits = output.logits[:, -1, :]
            limit = len(forced_ids) if forced_ids is not None else max_new_tokens
            for position in range(limit):
                if forced_ids is None:
                    token = _sample_token(torch, logits, temperature, generator)
                    token_id = int(token.item())
                    if token_id in stops:
                        break
                else:
                    token_id = int(forced_ids[position])
                    token = torch.tensor(
                        [token_id], dtype=prompt_ids.dtype, device=prompt_ids.device
                    )

                probability = float(
                    torch.softmax(logits.float(), dim=-1)[0, token_id].detach().cpu()
                )
                self.capture.reset()
                step = self.model(
                    input_ids=token.reshape(1, 1),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                reading = self.channels.score(
                    self.capture.take(self.detector_layer),
                    self.capture.take(self.subject_layer),
                )
                generated_ids.append(token_id)
                decoded = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                delta = (
                    decoded[len(previous_text) :]
                    if decoded.startswith(previous_text)
                    else decoded
                )
                previous_text = decoded
                yield TokenReading(
                    position=position,
                    token_id=token_id,
                    text=delta,
                    token_probability=probability,
                    channels=reading.as_dict(),
                )
                past = step.past_key_values
                logits = step.logits[:, -1, :]

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> Iterator[TokenReading]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        return self._step_stream(
            prompt,
            forced_ids=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )

    def replay(self, prompt: str, completion: str) -> Iterator[TokenReading]:
        encoded = self.tokenizer(
            completion,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        forced = [int(value) for value in encoded.tolist()]
        return self._step_stream(
            prompt,
            forced_ids=forced,
            max_new_tokens=len(forced),
            temperature=0.0,
            seed=0,
        )

    def trace_manifest(self, mode: str, prompt: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": mode,
            "prompt": prompt,
            "runtime": {
                **self.channels.manifest(),
                "quantization": self.quantization,
                "post_token_definition": "state after consuming each emitted non-EOS token",
                "generation_path": "incremental KV cache",
            },
        }


def write_jsonl_trace(
    path: str | Path, manifest: dict[str, Any], rows: list[TokenReading]
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "manifest", **manifest}, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(
                json.dumps({"type": "token", **row.as_dict()}, ensure_ascii=False) + "\n"
            )
