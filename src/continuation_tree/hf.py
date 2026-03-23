from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .core import PromptRecord


def _require_dependency(module_name: str, package_name: str) -> Any:
    try:
        return __import__(module_name, fromlist=["__name__"])
    except ImportError as exc:
        raise RuntimeError(
            f"{package_name} is required for Hugging Face integration but is not installed."
        ) from exc


def hf_dataset_to_prompts(
    dataset: Iterable[dict[str, Any]],
    *,
    prompt_id_column: str = "id",
    input_ids_column: str | None = "input_ids",
    text_column: str | None = "text",
    tokenizer_name: str | None = None,
    max_examples: int | None = None,
) -> list[PromptRecord]:
    prompts: list[PromptRecord] = []
    tokenizer = None
    if input_ids_column is None and text_column is None:
        raise ValueError("Either input_ids_column or text_column must be provided.")
    if input_ids_column is None:
        if tokenizer_name is None:
            raise ValueError("tokenizer_name is required when input_ids_column is not provided.")
        transformers = _require_dependency("transformers", "transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

    for index, row in enumerate(dataset):
        if max_examples is not None and index >= max_examples:
            break
        prompt_id = str(row.get(prompt_id_column, index))
        if input_ids_column is not None and row.get(input_ids_column) is not None:
            input_ids = tuple(int(token_id) for token_id in row[input_ids_column])
        else:
            encoded = tokenizer(row[text_column], add_special_tokens=False)
            input_ids = tuple(int(token_id) for token_id in encoded["input_ids"])
        prompts.append(PromptRecord(prompt_id=prompt_id, input_ids=input_ids))
    return prompts


@dataclass
class HFCausalLMBackend:
    model_name: str
    device: str = "cpu"
    dtype: str | None = None

    def __post_init__(self) -> None:
        torch = _require_dependency("torch", "torch")
        transformers = _require_dependency("transformers", "transformers")
        self._torch = torch
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        model_dtype = getattr(torch, self.dtype) if self.dtype else None
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=model_dtype,
        ).to(self.device)
        self._model.eval()

    def prefill_logits(self, input_ids: Sequence[int]) -> list[list[float]]:
        tensor = self._torch.tensor([list(input_ids)], dtype=self._torch.long, device=self.device)
        with self._torch.no_grad():
            logits = self._model(input_ids=tensor).logits[0, :-1, :]
        return logits.detach().cpu().tolist()

    def next_logits_batch(self, contexts: Sequence[Sequence[int]]) -> list[list[float]]:
        if not contexts:
            return []
        max_len = max(len(context) for context in contexts)
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id
        input_rows = []
        attention_rows = []
        for context in contexts:
            padding = [pad_id] * (max_len - len(context))
            input_rows.append(padding + list(context))
            attention_rows.append([0] * len(padding) + [1] * len(context))
        input_tensor = self._torch.tensor(input_rows, dtype=self._torch.long, device=self.device)
        attention_tensor = self._torch.tensor(attention_rows, dtype=self._torch.long, device=self.device)
        with self._torch.no_grad():
            logits = self._model(input_ids=input_tensor, attention_mask=attention_tensor).logits
        last_indices = attention_tensor.sum(dim=1) - 1
        rows = []
        for batch_idx, last_index in enumerate(last_indices.tolist()):
            rows.append(logits[batch_idx, last_index, :].detach().cpu().tolist())
        return rows

    def decode_token(self, token_id: int) -> str:
        return self._tokenizer.decode([token_id])
