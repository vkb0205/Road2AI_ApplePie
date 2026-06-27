"""IRAC answer generation for G-LRAG (CPU-import-safe, GPU-lazy).

Ties together :mod:`generation.prompt` (IRAC prompt) and
:mod:`generation.guardrails` (citation grounding + disclaimer) behind a small
:class:`IRACGenerator` that takes an **injected** ``llm_call`` callable. This
mirrors the dependency-injection style of :mod:`retrieval.doc_anchor`:

* The class imports only stdlib + the two pure generation helpers, so it
  imports and unit-tests on a CPU-only box with a fake ``llm_call``.
* The real Kaggle wiring builds an ``llm_call`` around a 4-bit Qwen causal LM
  via :func:`build_hf_llm_call` (which lazily imports torch / transformers).

A single model is loaded once and reused — no router/planner/generator triple
load, and no hardcoded law-id / domain tables anywhere in the path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from generation.guardrails import GuardrailConfig, apply_guardrails
from generation.prompt import PromptConfig, build_messages, render_plain

__all__ = [
    "GenerationConfig",
    "IRACGenerator",
    "build_hf_llm_call",
]

# Injected LLM: takes a chat message list, returns the raw generated string.
LLMCall = Callable[[List[Dict[str, str]]], str]


@dataclass
class GenerationConfig:
    """Top-level generation knobs (prompt + guardrails composed)."""

    prompt: PromptConfig = None  # type: ignore[assignment]
    guardrails: GuardrailConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.prompt is None:
            self.prompt = PromptConfig()
        if self.guardrails is None:
            self.guardrails = GuardrailConfig()


class IRACGenerator:
    """Generate a grounded IRAC answer from retrieved contexts.

    Parameters
    ----------
    llm_call:
        ``(messages) -> str``. ``messages`` is a chat ``[{role, content}]``
        list (from :func:`generation.prompt.build_messages`). May be ``None``
        to run in retrieval-only mode (``generate`` returns an empty answer),
        which is useful for a fast retrieval-only submission.
    cfg:
        :class:`GenerationConfig` controlling prompt + guardrails.
    """

    def __init__(
        self,
        llm_call: Optional[LLMCall] = None,
        cfg: Optional[GenerationConfig] = None,
    ) -> None:
        self.llm_call = llm_call
        self.cfg = cfg or GenerationConfig()

    def generate(
        self,
        question: str,
        contexts: Sequence[Dict[str, Any]],
        relevant_articles: Sequence[str],
    ) -> str:
        """Return a guardrail-checked IRAC answer string.

        When ``llm_call`` is ``None`` an empty string is returned (retrieval-
        only mode). Otherwise the model output is passed through the guardrails
        so the answer always cites an in-list ``Điều X`` (when one exists) and
        carries the reference-only disclaimer.
        """
        if self.llm_call is None:
            return ""
        messages = build_messages(question, contexts, relevant_articles, self.cfg.prompt)
        raw = self.llm_call(messages)
        answer, _report = apply_guardrails(
            raw or "", relevant_articles, self.cfg.guardrails
        )
        return answer


# --------------------------------------------------------------------------- #
# Real HF causal-LM call factory (lazy GPU imports)
# --------------------------------------------------------------------------- #
def build_hf_llm_call(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: Optional[str] = None,
    load_in_4bit: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_input_tokens: int = 3072,
    gpu_index: int = 0,
) -> LLMCall:
    """Build an ``llm_call`` around a (4-bit) HF causal LM. Lazy GPU imports.

    Loads the model exactly once and returns a closure that renders a chat
    message list (via the tokenizer's chat template when available) and
    generates a completion. NF4 4-bit loading keeps a 7B model near ~5 GB so it
    co-resides with the retrieval stack on one T4.
    """
    import torch  # lazy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if dev.startswith("cuda") and load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs.update(
            device_map={"": gpu_index},
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
        )
    elif dev.startswith("cuda"):
        load_kwargs.update(device_map={"": gpu_index}, torch_dtype=torch.float16)
    else:
        load_kwargs.update(torch_dtype=torch.float32)

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    def _llm_call(messages: List[Dict[str, str]]) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = render_plain(messages)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                repetition_penalty=1.02,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    return _llm_call
