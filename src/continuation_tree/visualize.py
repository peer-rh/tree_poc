from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from .core import AnchorTreeRow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render continuation-tree rows for one sample from a JSONL batch."
    )
    parser.add_argument("--input", required=True, help="Path to the generated JSONL file.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based sample index, counted over unique prompt_ids in file order.",
    )
    parser.add_argument(
        "--prompt-id",
        default=None,
        help="Prompt id to render. Overrides --sample-index when set.",
    )
    return parser


def row_from_dict(payload: dict[str, object]) -> AnchorTreeRow:
    return AnchorTreeRow(
        prompt_id=str(payload["prompt_id"]),
        anchor_token_index=int(payload["anchor_token_index"]),
        anchor_token_id=int(payload["anchor_token_id"]),
        observed_token_id=int(payload["observed_token_id"]),
        anchor_score=float(payload["anchor_score"]),
        edges=tuple((int(parent), int(child)) for parent, child in payload["edges"]),
        top_K=tuple(None if rank is None else int(rank) for rank in payload["top_K"]),
        node_token_ids=tuple(
            None if token_id is None else int(token_id) for token_id in payload["node_token_ids"]
        ),
        node_text=tuple(None if text is None else str(text) for text in payload["node_text"]),
        node_position_ids=tuple(int(position_id) for position_id in payload["node_position_ids"]),
        tree_spec_version=str(payload.get("tree_spec_version", "v1")),
    )


def load_rows(path: str | Path) -> list[AnchorTreeRow]:
    rows: list[AnchorTreeRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(row_from_dict(json.loads(line)))
    return rows


def select_prompt_rows(
    rows: Sequence[AnchorTreeRow],
    *,
    sample_index: int = 0,
    prompt_id: str | None = None,
) -> tuple[str, list[AnchorTreeRow]]:
    grouped: dict[str, list[AnchorTreeRow]] = {}
    for row in rows:
        grouped.setdefault(row.prompt_id, []).append(row)

    if not grouped:
        raise ValueError("No rows were found in the input batch.")

    if prompt_id is not None:
        if prompt_id not in grouped:
            available = ", ".join(grouped)
            raise ValueError(f"Unknown prompt_id {prompt_id!r}. Available prompt_ids: {available}")
        return prompt_id, grouped[prompt_id]

    prompt_ids = list(grouped)
    if sample_index < 0 or sample_index >= len(prompt_ids):
        raise IndexError(f"sample_index must be in [0, {len(prompt_ids) - 1}]")
    selected_prompt_id = prompt_ids[sample_index]
    return selected_prompt_id, grouped[selected_prompt_id]


def _node_children(row: AnchorTreeRow) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for parent_id, child_id in row.edges:
        children.setdefault(parent_id, []).append(child_id)
    return children


def _format_token_label(row: AnchorTreeRow, node_id: int) -> str:
    token_text = row.node_text[node_id]
    token_id = row.node_token_ids[node_id]
    token_repr = json.dumps(token_text, ensure_ascii=True) if token_text is not None else str(token_id)
    return (
        f"{token_repr} "
        f"(token_id={token_id}, rank={row.top_K[node_id]}, pos={row.node_position_ids[node_id]})"
    )


def _render_subtree(
    row: AnchorTreeRow,
    children: dict[int, list[int]],
    node_id: int,
    *,
    prefix: str,
    is_last: bool,
) -> list[str]:
    branch = "└── " if is_last else "├── "
    lines = [f"{prefix}{branch}{_format_token_label(row, node_id)}"]
    child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
    child_ids = children.get(node_id, [])
    for child_offset, child_id in enumerate(child_ids):
        lines.extend(
            _render_subtree(
                row,
                children,
                child_id,
                prefix=child_prefix,
                is_last=child_offset == len(child_ids) - 1,
            )
        )
    return lines


def render_anchor_tree(row: AnchorTreeRow) -> str:
    children = _node_children(row)
    lines = [
        (
            "anchor "
            f"idx={row.anchor_token_index} "
            f"anchor_token_id={row.anchor_token_id} "
            f"observed_token_id={row.observed_token_id} "
            f"score={row.anchor_score:.4f}"
        )
    ]
    root_children = children.get(0, [])
    for child_offset, child_id in enumerate(root_children):
        lines.extend(
            _render_subtree(
                row,
                children,
                child_id,
                prefix="",
                is_last=child_offset == len(root_children) - 1,
            )
        )
    return "\n".join(lines)


def render_sample(
    rows: Sequence[AnchorTreeRow],
    *,
    sample_index: int = 0,
    prompt_id: str | None = None,
) -> str:
    selected_prompt_id, selected_rows = select_prompt_rows(
        rows,
        sample_index=sample_index,
        prompt_id=prompt_id,
    )
    sections = [
        f"sample prompt_id={selected_prompt_id} anchors={len(selected_rows)}",
    ]
    for offset, row in enumerate(selected_rows):
        if offset > 0:
            sections.append("")
        sections.append(render_anchor_tree(row))
    return "\n".join(sections)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = load_rows(args.input)
    print(render_sample(rows, sample_index=args.sample_index, prompt_id=args.prompt_id))


if __name__ == "__main__":
    main()
