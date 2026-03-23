from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MockCausalLM:
    vocab: dict[int, str]
    prefill_map: dict[tuple[int, ...], list[list[float]]]
    next_map: dict[tuple[int, ...], list[float]]

    def prefill_logits(self, input_ids: Sequence[int]) -> list[list[float]]:
        key = tuple(input_ids)
        if key not in self.prefill_map:
            raise KeyError(f"Missing prefill logits for {key!r}")
        return self.prefill_map[key]

    def next_logits_batch(self, contexts: Sequence[Sequence[int]]) -> list[list[float]]:
        rows = []
        for context in contexts:
            key = tuple(context)
            if key not in self.next_map:
                raise KeyError(f"Missing next-token logits for {key!r}")
            rows.append(self.next_map[key])
        return rows

    def decode_token(self, token_id: int) -> str:
        return self.vocab[token_id]
