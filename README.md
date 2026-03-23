# continuation-tree

`continuation-tree` builds small continuation trees around high-surprise positions in tokenized prompts.
It scores each anchor position from causal language model prefill logits, forces the observed next token to remain the first root branch, and expands a fixed tree that can be serialized to JSONL or packed for flex-attention-style masking.

## What It Does

For each prompt, the library:

1. Runs a prefill pass over the prompt.
2. Scores every valid anchor position with `1 - p(observed_next_token)`.
3. Selects the top-scoring anchors.
4. Builds a fixed-width continuation tree for each anchor.

The default `TreeSpec` shape is:

- Root width: `3`
- Expanded root ranks: `(0, 1)`
- Children per expanded node: `2`

That produces this edge layout for each anchor:

```text
0
├── 1
│   ├── 4
│   └── 5
├── 2
│   ├── 6
│   └── 7
└── 3
```

Node `0` is the anchor root. Root child `1` is always the actually observed next token from the source prompt.

## Repository Layout

```text
src/continuation_tree/core.py        Core data types and tree generation
src/continuation_tree/hf.py          Hugging Face dataset and model integration
src/continuation_tree/flex.py        Packed frontier helpers and mask construction
src/continuation_tree/visualize.py   JSONL rendering utilities and sample viewer
tests/                               Coverage for core behavior, HF helpers, masks, and rendering
```

## Installation

The package itself has no required runtime dependencies declared beyond Python 3.11+:

```bash
python -m pip install -e .
```

Optional integrations require extra packages:

- Hugging Face dataset loading: `datasets`
- Tokenization and model loading: `transformers`
- Model execution and flex attention block masks: `torch`

Example:

```bash
python -m pip install -e . datasets transformers torch pytest
```

## Quick Start

This example uses the in-repo `MockCausalLM`, so it runs without external model dependencies.

```python
from continuation_tree import ContinuationTreeGenerator, MockCausalLM, PromptRecord

vocab = {
    0: "<pad>",
    1: "the",
    2: "fox",
    3: "jumps",
    4: "across",
    5: "over",
    6: "a",
    7: "the_det",
    8: "fence",
    9: "stream",
    10: "hill",
}

prefill = {
    (1, 2, 3, 4): [
        [0.0, 0.1, 3.4, 0.2, 0.3, 0.1, 0.0, -0.2, 0.2, 0.0, 0.0],
        [0.0, 0.2, 0.1, 2.0, 1.5, 1.4, 0.0, 0.0, -0.5, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0, 1.8, 2.6, 1.0, 0.2, -0.2, 0.1, 0.0],
    ],
}

next_map = {
    (1, 2, 3): [0.0, 0.0, 0.0, 0.0, 1.8, 2.6, 1.0, 0.2, -0.2, 0.1, 0.0],
    (1, 2, 3, 4): [0.0, 0.0, 0.0, 0.0, 0.1, -0.2, 1.4, 2.2, 1.7, 0.4, 0.3],
    (1, 2, 3, 5): [0.0, 0.0, 0.0, 0.0, -0.1, 0.1, 2.1, 1.8, 0.5, 1.9, 0.7],
}

model = MockCausalLM(vocab=vocab, prefill_map=prefill, next_map=next_map)
prompt = PromptRecord(prompt_id="p1", input_ids=(1, 2, 3, 4))

generator = ContinuationTreeGenerator(model)
anchor = generator.select_top_anchors(prompt, 1)[0]
row = generator.build_tree_for_anchor(prompt, anchor)

print(anchor.token_index)
print(row.edges)
print(row.to_json())
```

Expected highlights:

- The top anchor index is `2`.
- The root children are token ids `(4, 5, 6)`.
- The serialized edge list is `((0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (2, 7))`.

## Core API

Main exports:

- `PromptRecord`: tokenized prompt input
- `ContinuationTreeGenerator`: anchor scoring and tree construction
- `TreeSpec`: fixed tree layout definition
- `AnchorTreeRow`: JSON-serializable tree row
- `MockCausalLM`: lightweight backend for tests and examples
- `HFCausalLMBackend`: Hugging Face causal LM backend
- `hf_dataset_to_prompts`: dataset conversion helper
- `PackedFrontier`: packed-node representation for attention masks
- `build_attention_mask_matrix`: pure-Python mask builder

The backend contract is intentionally small:

- `prefill_logits(input_ids) -> [len(input_ids) - 1, vocab_size]`
- `next_logits_batch(contexts) -> [len(contexts), vocab_size]`
- `decode_token(token_id) -> str`

## Using Hugging Face Models

`HFCausalLMBackend` wraps a decoder-only model from `transformers`, and `hf_dataset_to_prompts` converts dataset rows into `PromptRecord` instances.

```python
from datasets import load_dataset

from continuation_tree import (
    ContinuationTreeGenerator,
    HFCausalLMBackend,
    hf_dataset_to_prompts,
)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5]")
prompts = hf_dataset_to_prompts(
    dataset,
    prompt_id_column="id",
    input_ids_column=None,
    text_column="text",
    tokenizer_name="gpt2",
)

backend = HFCausalLMBackend(model_name="gpt2", device="cpu")
generator = ContinuationTreeGenerator(backend)
rows = generator.build_rows(prompts, anchors_per_prompt=1)
generator.write_jsonl("artifacts/trees.jsonl", rows)
```

Notes:

- If your dataset already contains `input_ids`, no tokenizer is needed.
- If you pass raw text only, `tokenizer_name` is required.
- If your dataset has separate `prompt` and `response` fields, use `prompt_column` and `response_column`; the resulting `PromptRecord.anchor_start_index` is set so anchors are chosen only from the response segment.
- The package does not currently declare these optional dependencies in `pyproject.toml`, so install them manually.

## CLI Usage

Generate JSONL rows from a Hugging Face dataset:

```bash
python -m continuation_tree.cli \
  --dataset-path wikitext \
  --split train \
  --model gpt2 \
  --output artifacts/trees.jsonl \
  --anchors-per-prompt 1 \
  --max-examples 10 \
  --text-column text \
  --input-ids-column input_ids
```

Arguments of interest:

- `--dataset-path`: dataset name or local dataset path
- `--model`: causal LM name
- `--output`: output JSONL path
- `--anchors-per-prompt`: number of top anchors to keep
- `--tokenizer`: tokenizer override when tokenizing raw text
- `--device`: defaults to `cpu`
- `--dtype`: optional torch dtype name such as `float16` or `bfloat16`

To render one sample from a generated batch:

```bash
continuation-tree-show-sample --input artifacts/trees.jsonl --sample-index 0
```

Or select by prompt id:

```bash
continuation-tree-show-sample --input artifacts/trees.jsonl --prompt-id p1
```

For JSONL rows shaped like `{"prompt": "...", "response": "..."}`, load them through the Hugging Face JSON builder and restrict anchors to the response with:

```bash
python -m continuation_tree.cli \
  --dataset-path json \
  --data-files data/train.jsonl \
  --split train \
  --model gpt2 \
  --output artifacts/trees.jsonl \
  --prompt-column prompt \
  --response-column response \
  --text-joiner "" \
  --anchors-per-prompt 1
```

Example rendered output:

```text
sample prompt_id=p1 anchors=1
anchor idx=2 anchor_token_id=3 observed_token_id=4 score=0.6903
├── "across" (token_id=4, rank=0, pos=3)
│   ├── "the_det" (token_id=7, rank=0, pos=4)
│   └── "fence" (token_id=8, rank=1, pos=4)
├── "over" (token_id=5, rank=1, pos=3)
│   ├── "a" (token_id=6, rank=0, pos=4)
│   └── "stream" (token_id=9, rank=1, pos=4)
└── "a" (token_id=6, rank=2, pos=3)
```

## Flex Attention Packing

`PackedFrontier` stores flattened tree nodes with tree ids, parent links, depths, and position ids. `build_attention_mask_matrix` then creates a boolean matrix where a query can only attend to:

- Nodes from the same tree
- Nodes at positions not later than the query
- Ancestors in that tree

If `torch.nn.attention.flex_attention` is available, `build_block_mask` constructs a `create_block_mask(...)` object from the same ancestry rules.

## Development

Run the test suite with:

```bash
pytest
```

Current tests cover:

- Anchor scoring and selection
- Fixed-shape tree serialization
- Packed vs. naive tree expansion parity
- Hugging Face prompt conversion behavior
- Visualization rendering and selection
- Attention mask ancestry constraints
