"""Quantized LLM via turboquant-ml — same trick as ragforge-ml."""

from __future__ import annotations

from typing import Any

from agentforge.llm.hf import HFLLM


class QuantizedHFLLM(HFLLM):
    """:class:`HFLLM` that applies a TurboQuant method before serving.

    Example
    -------
    >>> llm = QuantizedHFLLM("meta-llama/Llama-3.2-3B-Instruct", method="bnb-nf4")
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        *,
        method: str = "bnb-nf4",
        quant_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.method = method
        self.model = _quantize(self.model, method=method, **(quant_kwargs or {}))


def _quantize(model, *, method: str, **kw):
    try:
        from turboquant import quantize
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "turboquant-ml is required for QuantizedHFLLM. "
            'Install with `pip install "agentforge-ml[quantized]"`.'
        ) from e
    return quantize(model, method=method, **kw)
