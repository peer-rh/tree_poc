from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ContinuationTreeGenerator
from .hf import HFCausalLMBackend, hf_dataset_to_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate continuation trees from a Hugging Face dataset.")
    parser.add_argument("--dataset-path", required=True, help="Dataset name or local dataset path.")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=None,
        help="Optional local files passed to datasets.load_dataset, for example JSONL shards.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", required=True, help="Causal LM model name.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--anchors-per-prompt", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--input-ids-column", default="input_ids")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--prompt-column", default=None)
    parser.add_argument("--response-column", default=None)
    parser.add_argument("--prompt-id-column", default="id")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer override when tokenizing raw text.")
    parser.add_argument(
        "--text-joiner",
        default="",
        help="Optional text inserted between prompt and response before the response anchor region begins.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    datasets = __import__("datasets", fromlist=["load_dataset"])
    load_dataset_kwargs = {"split": args.split}
    if args.data_files is not None:
        load_dataset_kwargs["data_files"] = args.data_files
    dataset = datasets.load_dataset(args.dataset_path, **load_dataset_kwargs)
    prompts = hf_dataset_to_prompts(
        dataset,
        prompt_id_column=args.prompt_id_column,
        input_ids_column=args.input_ids_column,
        text_column=args.text_column,
        prompt_column=args.prompt_column,
        response_column=args.response_column,
        tokenizer_name=args.tokenizer or args.model,
        text_joiner=args.text_joiner,
        max_examples=args.max_examples,
    )

    backend = HFCausalLMBackend(model_name=args.model, device=args.device, dtype=args.dtype)
    generator = ContinuationTreeGenerator(backend)
    rows = generator.build_rows(prompts, anchors_per_prompt=args.anchors_per_prompt)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("".join(f"{row.to_json()}\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
