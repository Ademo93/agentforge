"""Pretty-printable evaluation report."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalReport:
    n: int
    means: dict[str, float]
    per_sample: dict[str, list[float]]
    latencies_ms: list[float] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[int(0.95 * (len(s) - 1))]

    def as_table(self) -> str:
        width = max((len(m) for m in self.means), default=14)
        bar = "+" + "-" * (width + 4) + "+--------+"
        lines = [bar, f"| {'metric':<{width + 2}} | mean   |", bar]
        for name, val in self.means.items():
            lines.append(f"| {name:<{width + 2}} | {val:.3f} |")
        lines.append(bar)
        if self.latencies_ms:
            lines.append(f"n={self.n}  ·  p50={self.p50_ms:.0f}ms  ·  p95={self.p95_ms:.0f}ms")
        else:
            lines.append(f"n={self.n}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "n": self.n,
                    "means": self.means,
                    "per_sample": self.per_sample,
                    "latencies_ms": self.latencies_ms,
                    "extras": self.extras,
                },
                indent=2,
            )
        )
