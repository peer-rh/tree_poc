from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PackedNode:
    flat_index: int
    tree_id: int
    node_id: int
    parent_flat_index: int | None
    depth: int
    position_id: int


@dataclass(frozen=True)
class PackedFrontier:
    nodes: tuple[PackedNode, ...]

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[tuple[int, int, int | None, int, int]],
    ) -> "PackedFrontier":
        nodes = tuple(
            PackedNode(
                flat_index=flat_index,
                tree_id=tree_id,
                node_id=node_id,
                parent_flat_index=parent_flat_index,
                depth=depth,
                position_id=position_id,
            )
            for flat_index, (tree_id, node_id, parent_flat_index, depth, position_id) in enumerate(rows)
        )
        return cls(nodes=nodes)

    def ancestor_closure(self, flat_index: int) -> set[int]:
        ancestors = {flat_index}
        current = self.nodes[flat_index]
        while current.parent_flat_index is not None:
            ancestors.add(current.parent_flat_index)
            current = self.nodes[current.parent_flat_index]
        return ancestors


def build_attention_mask_matrix(frontier: PackedFrontier) -> list[list[bool]]:
    size = len(frontier.nodes)
    mask = [[False for _ in range(size)] for _ in range(size)]
    ancestor_cache = [frontier.ancestor_closure(i) for i in range(size)]
    for query in frontier.nodes:
        permitted = ancestor_cache[query.flat_index]
        for key in frontier.nodes:
            if query.tree_id != key.tree_id:
                continue
            if key.position_id > query.position_id:
                continue
            if key.flat_index in permitted:
                mask[query.flat_index][key.flat_index] = True
    return mask


def maybe_import_torch() -> tuple[object | None, object | None]:
    try:
        import torch
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        return None, None
    return torch, create_block_mask


def build_block_mask(frontier: PackedFrontier):
    torch, create_block_mask = maybe_import_torch()
    if torch is None or create_block_mask is None:
        raise RuntimeError("torch flex_attention is not installed in this environment.")

    tree_ids = torch.tensor([node.tree_id for node in frontier.nodes], dtype=torch.int64)
    position_ids = torch.tensor([node.position_id for node in frontier.nodes], dtype=torch.int64)
    ancestor_sets = [frontier.ancestor_closure(i) for i in range(len(frontier.nodes))]

    def mask_mod(batch_idx: int, head_idx: int, query_idx: int, key_idx: int) -> bool:
        del batch_idx, head_idx
        if tree_ids[query_idx] != tree_ids[key_idx]:
            return False
        if position_ids[key_idx] > position_ids[query_idx]:
            return False
        return int(key_idx) in ancestor_sets[int(query_idx)]

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=len(frontier.nodes),
        KV_LEN=len(frontier.nodes),
        device="cpu",
    )
