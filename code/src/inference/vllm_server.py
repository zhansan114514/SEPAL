"""
vLLM-based inference service for actor and critic models.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, Optional

from src.utils.generation_audit import (
    GenerationText,
    clone_generation_stats,
    empty_generation_stats,
    record_generation,
)
from src.utils.runtime_env import configure_runtime_libraries

logger = logging.getLogger(__name__)

MISTRAL_INFERENCE_TEMPLATE_VERSION = "mistral_inst_v2_single_bos"
MISTRAL_V03_INFERENCE_TEMPLATE_VERSION = "mistral_v03_token_ids_v2_single_bos"
GEMMA2_INFERENCE_TEMPLATE_VERSION = "gemma2_chat_template_v2_single_bos"
QWEN25_INFERENCE_TEMPLATE_VERSION = "qwen2_5_chat_template_v1"
QWEN3_INSTRUCT_2507_TEMPLATE_VERSION = "qwen3_instruct_2507_chat_template_v1"
PHI4_INFERENCE_TEMPLATE_VERSION = "phi4_tokenizer_chat_template_v1"
DEFAULT_INFERENCE_TEMPLATE_VERSION = "tokenizer_chat_template_v1"


def inference_prompt_format_version(
    model_type: str | None,
    model_name: str | None = None,
) -> str:
    """Return the generation template identity used for cache provenance."""
    identifier = model_type if model_type is not None else model_name
    normalized = str(identifier or "").strip().lower().replace("-", "_")
    normalized_name = str(model_name or "").strip().lower().replace("-", "_")
    if "qwen3" in normalized and "instruct_2507" in normalized_name:
        return QWEN3_INSTRUCT_2507_TEMPLATE_VERSION
    if (
        "qwen2.5" in normalized
        or "qwen2_5" in normalized
        or normalized == "qwen25"
        or normalized.startswith("qwen25_")
    ):
        return QWEN25_INFERENCE_TEMPLATE_VERSION
    if (
        normalized in {"mistral_v03", "mistral_v0.3", "mistral_v0_3"}
        or (model_type is None and "mistral" in normalized and "v0.3" in normalized)
    ):
        return MISTRAL_V03_INFERENCE_TEMPLATE_VERSION
    if "mistral" in normalized:
        return MISTRAL_INFERENCE_TEMPLATE_VERSION
    if "phi4" in normalized or "phi_4" in normalized:
        return PHI4_INFERENCE_TEMPLATE_VERSION
    basename = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    if (
        basename == "gemma2"
        or basename.startswith("gemma2_")
        or basename == "gemma_2"
        or basename.startswith("gemma_2_")
    ):
        return GEMMA2_INFERENCE_TEMPLATE_VERSION
    return DEFAULT_INFERENCE_TEMPLATE_VERSION


class VLLMInference:
    """Inference engine using vLLM for efficient LLM serving.

    Supports GPU placement via the ``cuda_device`` parameter.  Pass one
    physical GPU id for single-GPU inference, or a sequence of physical GPU ids
    for tensor-parallel inference across multiple GPUs.  vLLM's engine child
    process is restricted to those devices by temporarily overriding
    ``CUDA_VISIBLE_DEVICES``.

    LoRA support:
      Set ``enable_lora=True`` to allow dynamic LoRA adapter loading
      via ``generate_with_lora()``.  The engine will reserve GPU memory
      for up to ``max_loras`` adapters of rank ``max_lora_rank``.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 1024,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        cuda_device: Optional[int | Sequence[int] | str] = None,
        enable_lora: bool = False,
        max_loras: int = 1,
        max_lora_rank: int = 256,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        enforce_eager: bool = False,
        disable_custom_all_reduce: bool = False,
        load_format: str | None = None,
        safetensors_load_strategy: str | None = None,
        enable_prefix_caching: bool = True,
        enable_v1_multiprocessing: bool | None = None,
        language_model_only: bool = False,
        gdn_prefill_backend: str | None = None,
        compilation_config: dict[str, Any] | int | Any | None = None,
        disable_lora_cudagraph: bool = True,
        seed: int = 42,
        model_type: str | None = None,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self._max_model_len = int(max_model_len)
        if self._max_model_len < 2:
            raise ValueError(f"max_model_len must be >= 2, got {max_model_len}")
        self._llm = None
        self._tokenizer = None
        self._mistral_prompt_format_audited = False
        self._mistral_v03_prompt_format_audited = False
        self._gemma2_prompt_format_audited = False
        self._cuda_devices = self._normalize_cuda_devices(cuda_device)
        self._enable_lora = enable_lora
        self._enable_v1_multiprocessing = enable_v1_multiprocessing
        self._generation_stats = empty_generation_stats()
        if tensor_parallel_size is None:
            tensor_parallel_size = len(self._cuda_devices) if self._cuda_devices else 1
        if tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}"
            )
        if self._cuda_devices and tensor_parallel_size > len(self._cuda_devices):
            raise ValueError(
                "tensor_parallel_size cannot exceed number of selected CUDA devices: "
                f"tensor_parallel_size={tensor_parallel_size}, "
                f"devices={list(self._cuda_devices)}"
            )

        self._init_kwargs: dict = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": enforce_eager,
            "disable_log_stats": True,
            "enable_prefix_caching": enable_prefix_caching,
            "seed": int(seed),
        }
        if language_model_only:
            self._init_kwargs["language_model_only"] = True
        if gdn_prefill_backend:
            self._init_kwargs["gdn_prefill_backend"] = gdn_prefill_backend
        if disable_custom_all_reduce:
            self._init_kwargs["disable_custom_all_reduce"] = True
        if max_num_batched_tokens is not None:
            self._init_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if max_num_seqs is not None:
            self._init_kwargs["max_num_seqs"] = max_num_seqs
        if load_format:
            self._init_kwargs["load_format"] = load_format
        if safetensors_load_strategy:
            self._init_kwargs["safetensors_load_strategy"] = safetensors_load_strategy
        if enable_lora:
            self._init_kwargs["enable_lora"] = True
            self._init_kwargs["max_loras"] = max_loras
            self._init_kwargs["max_lora_rank"] = max_lora_rank
            logger.info(
                f"LoRA enabled: max_loras={max_loras}, max_lora_rank={max_lora_rank}"
            )
        if compilation_config is not None:
            self._init_kwargs["compilation_config"] = compilation_config
        elif enable_lora and disable_lora_cudagraph and not enforce_eager:
            self._init_kwargs["compilation_config"] = {"cudagraph_mode": "NONE"}
            logger.info("CUDA graph capture disabled for LoRA inference engine.")

    @staticmethod
    def _normalize_cuda_devices(
        cuda_device: Optional[int | Sequence[int] | str],
    ) -> Optional[tuple[int, ...]]:
        """Normalize a CUDA device spec into physical GPU ids."""
        if cuda_device is None:
            return None
        if isinstance(cuda_device, int):
            return (cuda_device,)
        if isinstance(cuda_device, str):
            parts = [part.strip() for part in cuda_device.split(",") if part.strip()]
            if not parts:
                return None
            return tuple(int(part) for part in parts)
        devices = tuple(int(device) for device in cuda_device)
        return devices or None

    def _ensure_loaded(self) -> None:
        """Lazy-load the vLLM engine on first use."""
        if self._llm is not None:
            return
        # Set CUDA_VISIBLE_DEVICES before any CUDA inspection or vLLM import.
        # cuda_device is always treated as a physical GPU ID.
        # Each vLLM instance spawns its own child process (EngineCore)
        # which inherits CUDA_VISIBLE_DEVICES at spawn time, so we can
        # safely change it between model loads without affecting already-
        # running engines.
        if self._cuda_devices is not None:
            target_physical = ",".join(str(device) for device in self._cuda_devices)
            os.environ["CUDA_VISIBLE_DEVICES"] = target_physical
            logger.info(
                f"Loading model on physical GPU(s) {target_physical}: "
                f"{self.model_name} "
                f"(tensor_parallel_size={self._init_kwargs['tensor_parallel_size']})"
            )
        else:
            logger.info(f"Loading model: {self.model_name}")

        # vLLM may spawn worker processes after this Python process has already
        # touched CUDA.  Forking in that state can fail with
        # "Cannot re-initialize CUDA in forked subprocess"; use spawn for the
        # engine workers.
        if self._enable_v1_multiprocessing is not None:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = (
                "1" if self._enable_v1_multiprocessing else "0"
            )
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        configure_runtime_libraries()
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM required. pip install vllm")

        # V100 (compute capability 7.0) does not support BF16
        # vLLM's dtype="auto" handles this automatically, but we add
        # explicit logging for clarity
        requested_dtype = self._init_kwargs.get("dtype", "auto")
        if requested_dtype == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    # Check first device (vLLM will use same dtype for all)
                    dev_name = torch.cuda.get_device_name(0)
                    if "V100" in dev_name or "TITAN V" in dev_name:
                        logger.info(
                            f"Detected {dev_name} (no BF16 support). "
                            "vLLM will use float16 automatically."
                        )
            except Exception:
                pass  # Silently skip if CUDA detection fails

        self._llm = LLM(self.model_name, **self._init_kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        # SoM/debate prompts put the current question and answer instructions at
        # the end.  If a peer response makes the prompt overflow, retain that
        # load-bearing suffix rather than the oldest peer-response prefix.
        self._configure_left_truncation(self._tokenizer)

    def generate(
        self,
        prompts: str | list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
        stop: Optional[list[str]] = None,
        seed: int | Sequence[int] | None = None,
        enable_thinking: bool | None = None,
    ) -> list[str]:
        """Generate text from prompts, optionally with one seed per request."""
        self._ensure_loaded()
        from vllm import SamplingParams

        if isinstance(prompts, str):
            prompts = [prompts]
        prompts, tokenization_kwargs = self._prepare_generation_prompts(
            prompts,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

        sampling_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop": stop,
        }
        params = self._sampling_params_for_requests(
            SamplingParams,
            request_count=len(prompts),
            sampling_kwargs=sampling_kwargs,
            seed=seed,
        )
        outputs = self._llm.generate(
            prompts,
            params,
            **tokenization_kwargs,
        )
        return self._record_outputs(outputs, max_tokens=max_tokens)

    @staticmethod
    def _configure_left_truncation(tokenizer: Any) -> None:
        """Use suffix-preserving truncation without mutating read-only tokenizers."""
        current = getattr(tokenizer, "truncation_side", None)
        if current == "left":
            return
        try:
            tokenizer.truncation_side = "left"
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "Tokenizer does not support the required left-side truncation: "
                f"type={type(tokenizer).__name__}, truncation_side={current!r}"
            ) from exc

    def _prepare_generation_prompts(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        enable_thinking: bool | None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Return vLLM prompts plus only the kwargs valid for their prompt type."""
        if self._uses_mistral_v03_chat_template():
            return self._tokenize_mistral_v03_prompts(prompts, max_tokens=max_tokens), {}
        formatted = self._format_prompts_for_thinking(prompts, enable_thinking)
        return formatted, {
            "tokenization_kwargs": self._generation_tokenization_kwargs(max_tokens)
        }

    def _tokenize_mistral_v03_prompts(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
    ) -> list[dict[str, list[int]]]:
        """Tokenize v0.3 chats once with Tokenizer V3 and pass IDs to vLLM."""
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise RuntimeError(
                "Mistral v0.3 inference requires its tokenizer chat template"
            )
        max_input_tokens = self._generation_tokenization_kwargs(max_tokens)["max_length"]
        prepared: list[dict[str, list[int]]] = []
        for prompt in prompts:
            raw = self._mistral_v03_user_content(prompt)
            token_ids = apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, dict):
                token_ids = token_ids.get("input_ids")
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or not all(isinstance(token_id, int) for token_id in token_ids)
            ):
                raise RuntimeError(
                    "Mistral v0.3 chat template did not return a flat token-id list"
                )
            if len(token_ids) > max_input_tokens:
                prefix_tokens = min(2, max_input_tokens)
                suffix_tokens = max_input_tokens - prefix_tokens
                token_ids = token_ids[:prefix_tokens] + (
                    token_ids[-suffix_tokens:] if suffix_tokens else []
                )
            self._audit_mistral_v03_token_ids(token_ids)
            prepared.append({"prompt_token_ids": token_ids})
        self._mistral_v03_prompt_format_audited = True
        return prepared

    def _audit_mistral_v03_token_ids(self, token_ids: list[int]) -> None:
        """Reject duplicate BOS or a terminal EOS before generation starts."""
        bos_id = getattr(self._tokenizer, "bos_token_id", None)
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        if bos_id is not None and (
            token_ids[0] != bos_id or token_ids.count(bos_id) != 1
        ):
            raise RuntimeError(
                "Mistral v0.3 tokenized prompt must contain exactly one leading BOS"
            )
        if eos_id is not None and token_ids[-1] == eos_id:
            raise RuntimeError(
                "Mistral v0.3 generation prompt must not end with EOS"
            )

    def _generation_tokenization_kwargs(self, max_tokens: int) -> dict[str, Any]:
        """Reserve context for completion and safely truncate oversized prompts."""
        max_tokens = int(max_tokens)
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        max_input_tokens = self._max_model_len - max_tokens
        if max_input_tokens < 1:
            raise ValueError(
                "max_tokens must leave room for at least one input token: "
                f"max_tokens={max_tokens}, max_model_len={self._max_model_len}"
            )
        # vLLM derives its input/output validation budget from max_length and
        # applies `truncation=True` before validating the total context length.
        return {"max_length": max_input_tokens, "truncation": True}

    @staticmethod
    def _sampling_params_for_requests(
        sampling_params_type: Any,
        *,
        request_count: int,
        sampling_kwargs: dict[str, Any],
        seed: int | Sequence[int] | None,
    ) -> Any:
        """Build scalar or per-request vLLM sampling parameters.

        vLLM accepts either one ``SamplingParams`` object for the entire batch
        or a sequence containing one object per request.  The latter is
        load-bearing for Monte Carlo rollouts that repeat an identical prompt:
        sharing a scalar seed can otherwise make the nominally independent
        draws identical.
        """
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes, bytearray)):
            seeds = [int(item) for item in seed]
            if len(seeds) != request_count:
                raise ValueError(
                    "Per-request seed count must match prompt count: "
                    f"{len(seeds)} != {request_count}"
                )
            return [
                sampling_params_type(**sampling_kwargs, seed=request_seed)
                for request_seed in seeds
            ]

        kwargs = dict(sampling_kwargs)
        if seed is not None:
            kwargs["seed"] = int(seed)
        return sampling_params_type(**kwargs)

    def _format_prompts_for_thinking(
        self,
        prompts: list[str],
        enable_thinking: bool | None,
    ) -> list[str]:
        """Apply model-native thinking controls when explicitly requested."""
        if not prompts:
            return prompts

        # Match the released ACC-Collab Mistral path exactly: vLLM receives an
        # ``[INST] ... [/INST]`` string without a rendered ``<s>`` token, then
        # adds one BOS while tokenizing the request.  Mistral's tokenizer chat
        # template already renders ``<s>``; passing that string to vLLM made it
        # add a second BOS and put preference-data inference out of alignment
        # with both the paper implementation and our SFT/DPO formatter.
        if self._uses_mistral_v03_chat_template():
            # Generation uses ``_tokenize_mistral_v03_prompts``.  Returning only
            # user content here prevents callers from accidentally taking the
            # unsafe tokenize=False -> encode path warned about by vLLM.
            return [self._mistral_v03_user_content(prompt) for prompt in prompts]

        if self._uses_mistral_instruction_template():
            formatted = [
                self._format_mistral_instruction_prompt(prompt) for prompt in prompts
            ]
            self._audit_mistral_single_bos(formatted[0])
            return formatted

        # Gemma 2's tokenizer chat template also renders its own ``<bos>``.
        # Strip that textual BOS before handing the string to vLLM, which adds
        # exactly one special BOS during tokenization.  Keep this branch
        # Gemma-2-specific so Llama and other model families are unchanged.
        if self._uses_gemma2_chat_template():
            formatted = [self._format_gemma2_chat_prompt(prompt) for prompt in prompts]
            self._audit_gemma2_single_bos(formatted[0])
            return formatted

        # Qwen2.5-Instruct owns a non-thinking ChatML template. Render it
        # directly instead of probing with the Qwen3-only ``enable_thinking``
        # keyword, keeping inference aligned with SFT and DPO formatting.
        if self._uses_qwen25_chat_template():
            return [self._format_qwen25_chat_prompt(prompt) for prompt in prompts]

        # Qwen3-Instruct-2507 is non-thinking-only. Its released template does
        # not expose the older ``enable_thinking`` switch, so render the native
        # ChatML prompt without passing that unsupported keyword.
        if self._uses_qwen3_instruct_2507_chat_template():
            return [self._format_qwen3_instruct_prompt(prompt) for prompt in prompts]

        if self._uses_phi4_chat_template():
            return [self._format_phi4_chat_prompt(prompt) for prompt in prompts]

        if enable_thinking is None:
            return prompts

        tokenizer = self._tokenizer
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            formatted: list[str] = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                try:
                    formatted.append(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking,
                        )
                    )
                    continue
                except TypeError:
                    # Llama and most non-reasoning tokenizers do not accept an
                    # ``enable_thinking`` keyword.  Apply their normal chat
                    # template instead of injecting Qwen-specific /no_think text.
                    try:
                        formatted.append(
                            tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        )
                        continue
                    except Exception as e:  # pragma: no cover - tokenizer/model specific
                        logger.warning("Chat template fallback failed; using raw prompt: %s", e)
                except Exception as e:  # pragma: no cover - tokenizer/model specific
                    logger.warning("Thinking chat template failed; using raw prompt: %s", e)
                formatted.append(prompt)
            return formatted

        return [
            self._append_thinking_control(prompt, enable_thinking)
            for prompt in prompts
        ]

    def _uses_mistral_instruction_template(self) -> bool:
        """Identify Mistral explicitly, with the model path as a compatibility fallback."""
        return inference_prompt_format_version(
            getattr(self, "model_type", None),
            self.model_name,
        ) == MISTRAL_INFERENCE_TEMPLATE_VERSION

    def _format_mistral_instruction_prompt(self, prompt: str) -> str:
        """Return the BOS-free Mistral wrapper expected by vLLM string prompts."""
        raw = self._strip_rendered_bos(prompt).strip()
        if raw.startswith("[INST]") and raw.endswith("[/INST]"):
            return raw
        return f"[INST] {raw} [/INST]"

    def _uses_mistral_v03_chat_template(self) -> bool:
        """Match Mistral v0.3 without changing the validated v0.2 wrapper."""
        return inference_prompt_format_version(
            getattr(self, "model_type", None),
            self.model_name,
        ) == MISTRAL_V03_INFERENCE_TEMPLATE_VERSION

    @staticmethod
    def _mistral_v03_user_content(prompt: str) -> str:
        """Return raw user content, unwrapping an already-rendered INST prompt."""
        original = str(prompt or "")
        normalized = original.strip()
        if normalized.startswith("[INST]") and normalized.endswith("[/INST]"):
            content = normalized[len("[INST]") : -len("[/INST]")]
            return content[1:] if content.startswith(" ") else content
        return original

    def _uses_gemma2_chat_template(self) -> bool:
        """Match Gemma 2 explicitly, with the model path as a compatibility fallback."""
        return inference_prompt_format_version(
            getattr(self, "model_type", None),
            self.model_name,
        ) == GEMMA2_INFERENCE_TEMPLATE_VERSION

    def _uses_qwen3_instruct_2507_chat_template(self) -> bool:
        """Match the non-thinking Qwen3-Instruct-2507 release explicitly."""
        return inference_prompt_format_version(
            getattr(self, "model_type", None),
            self.model_name,
        ) == QWEN3_INSTRUCT_2507_TEMPLATE_VERSION

    def _uses_qwen25_chat_template(self) -> bool:
        """Match Qwen2.5-Instruct explicitly, with model path as a fallback."""
        return inference_prompt_format_version(
            getattr(self, "model_type", None),
            self.model_name,
        ) == QWEN25_INFERENCE_TEMPLATE_VERSION

    def _uses_phi4_chat_template(self) -> bool:
        """Match Phi-4 explicitly so training and inference share one template."""
        return inference_prompt_format_version(
            getattr(self, "model_type", None),
            self.model_name,
        ) == PHI4_INFERENCE_TEMPLATE_VERSION

    def _format_qwen25_chat_prompt(self, prompt: str) -> str:
        """Render one Qwen2.5-Instruct user turn with its native tokenizer."""
        original = str(prompt or "")
        raw = original.strip()
        if "<|im_start|>user\n" in raw and "<|im_start|>assistant" in raw:
            return original
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise RuntimeError("Qwen2.5 inference requires its tokenizer chat template")
        return str(
            apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _format_qwen3_instruct_prompt(self, prompt: str) -> str:
        """Render one Qwen3-Instruct-2507 user turn with its native tokenizer."""
        original = str(prompt or "")
        raw = original.strip()
        if "<|im_start|>user\n" in raw and "<|im_start|>assistant" in raw:
            return original
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise RuntimeError(
                "Qwen3-Instruct-2507 inference requires its tokenizer chat template"
            )
        return str(
            apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _format_phi4_chat_prompt(self, prompt: str) -> str:
        """Render one Phi-4-mini user turn with the released tokenizer template."""
        original = str(prompt or "")
        raw = original.strip()
        if "<|user|>" in raw and "<|assistant|>" in raw:
            return original
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise RuntimeError("Phi-4 inference requires its tokenizer chat template")
        return str(
            apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _format_gemma2_chat_prompt(self, prompt: str) -> str:
        """Render Gemma 2 chat controls while leaving BOS insertion to vLLM."""
        raw = self._strip_rendered_bos(prompt)
        if "<start_of_turn>user\n" in raw and "<start_of_turn>model\n" in raw:
            return raw

        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise RuntimeError("Gemma 2 inference requires its tokenizer chat template")
        rendered = apply_chat_template(
            [{"role": "user", "content": raw}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self._strip_rendered_bos(str(rendered))

    def _strip_rendered_bos(self, prompt: str) -> str:
        """Remove textual BOS tokens before vLLM string-prompt tokenization."""
        raw = str(prompt or "").lstrip()
        bos_token = str(getattr(self._tokenizer, "bos_token", "") or "")
        while bos_token and raw.startswith(bos_token):
            raw = raw[len(bos_token) :].lstrip()
        return raw

    def _audit_mistral_single_bos(self, prompt: str) -> None:
        """Fail fast if vLLM's normal tokenization would create multiple BOS tokens."""
        self._audit_single_bos(
            prompt,
            family="Mistral",
            template_version=MISTRAL_INFERENCE_TEMPLATE_VERSION,
            state_attribute="_mistral_prompt_format_audited",
        )

    def _audit_gemma2_single_bos(self, prompt: str) -> None:
        """Fail fast if Gemma 2 string tokenization would create multiple BOS tokens."""
        self._audit_single_bos(
            prompt,
            family="Gemma 2",
            template_version=GEMMA2_INFERENCE_TEMPLATE_VERSION,
            state_attribute="_gemma2_prompt_format_audited",
        )

    def _audit_single_bos(
        self,
        prompt: str,
        *,
        family: str,
        template_version: str,
        state_attribute: str,
    ) -> None:
        """Audit one model-specific BOS-free string before its first request."""
        if getattr(self, state_attribute, False):
            return

        tokenizer = self._tokenizer
        bos_token = str(getattr(tokenizer, "bos_token", "") or "")
        if bos_token and prompt.lstrip().startswith(bos_token):
            raise RuntimeError(
                f"{family} prompt must not contain a rendered BOS before vLLM tokenization"
            )

        encode = getattr(tokenizer, "encode", None)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if callable(encode) and bos_token_id is not None:
            token_ids = encode(prompt, add_special_tokens=True)
            leading_bos = 0
            for token_id in token_ids:
                if int(token_id) != int(bos_token_id):
                    break
                leading_bos += 1
            if leading_bos != 1:
                raise RuntimeError(
                    f"{family} prompt single-BOS audit failed: "
                    f"expected 1 leading BOS token, got {leading_bos}"
                )

        setattr(self, state_attribute, True)
        logger.info(
            "%s inference prompt audit passed: template=%s, rendered_bos=false",
            family,
            template_version,
        )

    @staticmethod
    def _append_thinking_control(prompt: str, enable_thinking: bool) -> str:
        control = "/think" if enable_thinking else "/no_think"
        if control.lower() in prompt.lower():
            return prompt
        return f"{prompt.rstrip()}\n\n{control}"

    def _record_outputs(
        self,
        outputs: Sequence[Any],
        *,
        max_tokens: int | Sequence[int],
    ) -> list[str]:
        """Convert vLLM outputs to strings while retaining finish metadata."""
        limits = (
            [int(item) for item in max_tokens]
            if not isinstance(max_tokens, int)
            else [int(max_tokens)] * len(outputs)
        )
        if len(limits) != len(outputs):
            raise ValueError(
                "max-token limits must match vLLM request outputs: "
                f"{len(limits)} != {len(outputs)}"
            )
        texts: list[str] = []
        for request_index, request_output in enumerate(outputs):
            limit = limits[request_index]
            for choice in request_output.outputs:
                finish_reason = getattr(choice, "finish_reason", None)
                token_ids = getattr(choice, "token_ids", None)
                generated_tokens = len(token_ids) if token_ids is not None else None
                record_generation(
                    self._generation_stats,
                    max_tokens=limit,
                    finish_reason=finish_reason,
                    generated_tokens=generated_tokens,
                )
                texts.append(
                    GenerationText(
                        choice.text,
                        finish_reason=finish_reason,
                        generated_tokens=generated_tokens,
                        max_tokens=limit,
                    )
                )
        return texts

    def generation_stats(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot used by per-batch checkpoints."""
        return clone_generation_stats(self._generation_stats)

    @property
    def supports_lora(self) -> bool:
        """Whether this engine was initialized with LoRA support."""
        return self._enable_lora

    @property
    def tokenizer(self):
        """Underlying HF tokenizer (None until the engine is loaded)."""
        return self._tokenizer

    def generate_with_lora(
        self,
        prompts: str | list[str],
        lora_request: Any,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
        stop: Optional[list[str]] = None,
        seed: int | Sequence[int] | None = None,
        enable_thinking: bool | None = None,
    ) -> list[str]:
        """Generate text using a specific LoRA adapter.

        Raises RuntimeError if LoRA is not enabled on this engine.
        """
        if not self._enable_lora:
            raise RuntimeError(
                "LoRA is not enabled on this VLLMInference instance. "
                "Re-create with enable_lora=True to use LoRA adapters."
            )
        self._ensure_loaded()
        from vllm import SamplingParams

        if isinstance(prompts, str):
            prompts = [prompts]
        prompts, tokenization_kwargs = self._prepare_generation_prompts(
            prompts,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

        sampling_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop": stop,
        }
        params = self._sampling_params_for_requests(
            SamplingParams,
            request_count=len(prompts),
            sampling_kwargs=sampling_kwargs,
            seed=seed,
        )
        outputs = self._llm.generate(
            prompts,
            params,
            lora_request=lora_request,
            **tokenization_kwargs,
        )
        return self._record_outputs(outputs, max_tokens=max_tokens)

    def generate_with_loras(
        self,
        prompts: list[str],
        lora_requests: Sequence[Any],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
        stop: Optional[list[str]] = None,
        seed: int | Sequence[int] | None = None,
        enable_thinking: bool | None = None,
    ) -> list[str]:
        """Generate text with a per-prompt LoRA adapter request."""
        if not self._enable_lora:
            raise RuntimeError(
                "LoRA is not enabled on this VLLMInference instance. "
                "Re-create with enable_lora=True to use LoRA adapters."
            )
        if len(prompts) != len(lora_requests):
            raise ValueError(
                "prompts and lora_requests must have the same length: "
                f"{len(prompts)} != {len(lora_requests)}"
            )
        self._ensure_loaded()
        from vllm import SamplingParams
        prompts, tokenization_kwargs = self._prepare_generation_prompts(
            prompts,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

        sampling_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop": stop,
        }
        params = self._sampling_params_for_requests(
            SamplingParams,
            request_count=len(prompts),
            sampling_kwargs=sampling_kwargs,
            seed=seed,
        )
        outputs = self._llm.generate(
            prompts,
            params,
            lora_request=list(lora_requests),
            **tokenization_kwargs,
        )
        return self._record_outputs(outputs, max_tokens=max_tokens)

    def cleanup(self) -> None:
        """Explicitly release GPU memory."""
        if self._llm is not None:
            try:
                import gc
                import torch
                del self._llm
                del self._tokenizer
                self._llm = None
                self._tokenizer = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Cleaned up model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Cleanup failed for {self.model_name}: {e}")

    def __del__(self) -> None:
        """Destructor to ensure GPU memory is released."""
        self.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False

    def __repr__(self) -> str:
        return f"VLLMInference(model={self.model_name})"


def build_inference_engine(
    args,
    *,
    enable_lora: bool = False,
    max_loras: int | None = None,
    max_lora_rank: int | None = None,
    disable_lora_cudagraph: bool = False,
) -> VLLMInference:
    """Build a single-GPU ``VLLMInference`` from an experiment args namespace.

    ``args`` is the ``_vllm_args()`` projection of the experiment config
    (``model_name``, ``device``, ``dtype``, runtime knobs, and — for LoRA stages
    — the adapter fields). Every generation stage constructs its engine this
    same way, so the construction lives here once instead of being copied into
    each script. Pass LoRA overrides only when ``enable_lora`` is true.
    """

    def _opt_int(name: str) -> Optional[int]:
        value = getattr(args, name, None)
        return int(value) if value is not None else None

    init_kwargs: dict[str, Any] = {
        "cuda_device": args.device,
        "dtype": args.dtype,
        "seed": int(args.seed),
        "model_type": getattr(args, "model_type", None),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "enable_prefix_caching": bool(args.enable_prefix_caching),
        "max_num_batched_tokens": _opt_int("max_num_batched_tokens"),
        "max_num_seqs": _opt_int("max_num_seqs"),
        "enforce_eager": bool(args.enforce_eager),
        "language_model_only": bool(args.language_model_only),
        "gdn_prefill_backend": args.gdn_prefill_backend,
    }
    if enable_lora:
        init_kwargs.update(
            enable_lora=True,
            max_loras=int(max_loras),
            max_lora_rank=int(max_lora_rank),
            disable_lora_cudagraph=bool(disable_lora_cudagraph),
        )
    return VLLMInference(args.model_name, **init_kwargs)
