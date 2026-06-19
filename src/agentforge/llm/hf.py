"""HuggingFace causal LM backend.

ReAct-friendly: applies the model's chat template, accepts stop strings,
and truncates the output at the first stop marker so the next loop iteration
gets a clean continuation.
"""

from __future__ import annotations

from typing import Any

import torch


class HFLLM:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        *,
        dtype: str = "auto",
        device_map: str | dict | None = "auto",
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        text = self._apply_chat_template(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_ids = out[0, inputs.input_ids.shape[-1] :]
        answer = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return _truncate_at_stop(answer, stop)

    def _apply_chat_template(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt


def _truncate_at_stop(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text.strip()
    earliest = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            earliest = min(earliest, idx)
    return text[:earliest].strip()
