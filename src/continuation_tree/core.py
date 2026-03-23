from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Protocol, Sequence


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    input_ids: tuple[int, ...]
    tokens: tuple[str, ...] | None = None

    def token_text(self, token_id: int) -> str:
        if self.tokens is None:
            return str(token_id)
        try:
            idx = self.input_ids.index(token_id)
        except ValueError:
            return str(token_id)
        return self.tokens[idx]


@dataclass(frozen=True)
class AnchorCandidate:
    token_index: int
    anchor_token_id: int
    observed_token_id: int
    score: float


@dataclass(frozen=True)
class TreeNode:
    node_id: int
    parent_id: int | None
    token_id: int | None
    token_text: str | None
    top_k_rank: int | None
    depth: int
    position_id: int


@dataclass(frozen=True)
class AnchorTreeRow:
    prompt_id: str
    anchor_token_index: int
    anchor_token_id: int
    observed_token_id: int
    anchor_score: float
    edges: tuple[tuple[int, int], ...]
    top_K: tuple[int | None, ...]
    node_token_ids: tuple[int | None, ...]
    node_text: tuple[str | None, ...]
    node_position_ids: tuple[int, ...]
    tree_spec_version: str = "v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True)


@dataclass(frozen=True)
class TreeSpec:
    root_width: int = 3
    expanded_root_ranks: tuple[int, ...] = (0, 1)
    child_width: int = 2

    def expected_edges(self) -> tuple[tuple[int, int], ...]:
        edges: list[tuple[int, int]] = []
        node_id = 1
        root_children: dict[int, int] = {}
        for local_rank in range(self.root_width):
            edges.append((0, node_id))
            root_children[local_rank] = node_id
            node_id += 1
        for local_rank in self.expanded_root_ranks:
            parent_id = root_children[local_rank]
            for _ in range(self.child_width):
                edges.append((parent_id, node_id))
                node_id += 1
        return tuple(edges)


class CausalLMBackend(Protocol):
    def prefill_logits(self, input_ids: Sequence[int]) -> list[list[float]]:
        """
        Returns logits with shape [len(input_ids) - 1, vocab_size].
        Row t predicts input_ids[t + 1] from the prefix ending at input_ids[t].
        """

    def next_logits_batch(self, contexts: Sequence[Sequence[int]]) -> list[list[float]]:
        """Returns logits with shape [len(contexts), vocab_size]."""

    def decode_token(self, token_id: int) -> str:
        """Returns a human-readable token string."""


def _stable_descending_indices(values: Sequence[float]) -> list[int]:
    return [idx for idx, _ in sorted(enumerate(values), key=lambda item: (-item[1], item[0]))]


def _softmax(logits: Sequence[float]) -> list[float]:
    max_logit = max(logits)
    exp_values = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def topk_excluding(
    probs: Sequence[float],
    *,
    excluded_ids: set[int],
    k: int,
) -> list[tuple[int, int]]:
    picks: list[tuple[int, int]] = []
    for rank, token_id in enumerate(_stable_descending_indices(probs)):
        token_id = int(token_id)
        if token_id in excluded_ids:
            continue
        picks.append((token_id, rank))
        if len(picks) == k:
            return picks
    raise ValueError("Not enough candidate tokens to satisfy top-k selection.")


class ContinuationTreeGenerator:
    def __init__(self, model: CausalLMBackend, tree_spec: TreeSpec | None = None):
        self.model = model
        self.tree_spec = tree_spec or TreeSpec()

    def score_anchors(self, prompt: PromptRecord) -> list[AnchorCandidate]:
        if len(prompt.input_ids) < 2:
            return []
        logits = self.model.prefill_logits(prompt.input_ids)
        if len(logits) != len(prompt.input_ids) - 1:
            raise ValueError("prefill_logits must return one row per valid anchor position.")
        candidates: list[AnchorCandidate] = []
        for token_index in range(len(prompt.input_ids) - 1):
            probs = _softmax(logits[token_index])
            observed_token_id = prompt.input_ids[token_index + 1]
            score = float(1.0 - probs[observed_token_id])
            candidates.append(
                AnchorCandidate(
                    token_index=token_index,
                    anchor_token_id=prompt.input_ids[token_index],
                    observed_token_id=observed_token_id,
                    score=score,
                )
            )
        return candidates

    def select_top_anchors(self, prompt: PromptRecord, k: int) -> list[AnchorCandidate]:
        candidates = self.score_anchors(prompt)
        return sorted(candidates, key=lambda c: (-c.score, c.token_index))[:k]

    def build_tree_for_anchor(self, prompt: PromptRecord, anchor: AnchorCandidate) -> AnchorTreeRow:
        prompt_prefix = list(prompt.input_ids[: anchor.token_index + 1])
        root_logits = self.model.next_logits_batch([prompt_prefix])[0]
        root_probs = _softmax(root_logits)
        root_children = self._root_children(anchor.observed_token_id, root_probs)

        nodes: list[TreeNode] = [
            TreeNode(
                node_id=0,
                parent_id=None,
                token_id=None,
                token_text=None,
                top_k_rank=None,
                depth=0,
                position_id=anchor.token_index,
            )
        ]
        edges: list[tuple[int, int]] = []

        for child_rank, child_token_id in enumerate(root_children):
            node_id = len(nodes)
            edges.append((0, node_id))
            nodes.append(
                TreeNode(
                    node_id=node_id,
                    parent_id=0,
                    token_id=child_token_id,
                    token_text=self.model.decode_token(child_token_id),
                    top_k_rank=child_rank,
                    depth=1,
                    position_id=anchor.token_index + 1,
                )
            )

        self._append_depth_two_nodes(
            prompt_prefix=prompt_prefix,
            anchor_token_index=anchor.token_index,
            nodes=nodes,
            edges=edges,
        )

        return AnchorTreeRow(
            prompt_id=prompt.prompt_id,
            anchor_token_index=anchor.token_index,
            anchor_token_id=anchor.anchor_token_id,
            observed_token_id=anchor.observed_token_id,
            anchor_score=anchor.score,
            edges=tuple(edges),
            top_K=tuple(node.top_k_rank for node in nodes),
            node_token_ids=tuple(node.token_id for node in nodes),
            node_text=tuple(node.token_text for node in nodes),
            node_position_ids=tuple(node.position_id for node in nodes),
        )

    def build_tree_for_anchor_naive(self, prompt: PromptRecord, anchor: AnchorCandidate) -> AnchorTreeRow:
        prompt_prefix = list(prompt.input_ids[: anchor.token_index + 1])
        root_logits = self.model.next_logits_batch([prompt_prefix])[0]
        root_probs = _softmax(root_logits)
        root_children = self._root_children(anchor.observed_token_id, root_probs)

        nodes: list[TreeNode] = [
            TreeNode(
                node_id=0,
                parent_id=None,
                token_id=None,
                token_text=None,
                top_k_rank=None,
                depth=0,
                position_id=anchor.token_index,
            )
        ]
        edges: list[tuple[int, int]] = []
        for child_rank, child_token_id in enumerate(root_children):
            node_id = len(nodes)
            edges.append((0, node_id))
            nodes.append(
                TreeNode(
                    node_id=node_id,
                    parent_id=0,
                    token_id=child_token_id,
                    token_text=self.model.decode_token(child_token_id),
                    top_k_rank=child_rank,
                    depth=1,
                    position_id=anchor.token_index + 1,
                )
            )

        for local_rank in self.tree_spec.expanded_root_ranks:
            parent_id = local_rank + 1
            logits = self.model.next_logits_batch([prompt_prefix + [nodes[parent_id].token_id]])[0]
            self._append_children_from_logits(
                parent_id=parent_id,
                logits=logits,
                anchor_token_index=anchor.token_index,
                nodes=nodes,
                edges=edges,
            )

        return AnchorTreeRow(
            prompt_id=prompt.prompt_id,
            anchor_token_index=anchor.token_index,
            anchor_token_id=anchor.anchor_token_id,
            observed_token_id=anchor.observed_token_id,
            anchor_score=anchor.score,
            edges=tuple(edges),
            top_K=tuple(node.top_k_rank for node in nodes),
            node_token_ids=tuple(node.token_id for node in nodes),
            node_text=tuple(node.token_text for node in nodes),
            node_position_ids=tuple(node.position_id for node in nodes),
        )

    def build_rows(self, prompts: Iterable[PromptRecord], anchors_per_prompt: int) -> list[AnchorTreeRow]:
        rows: list[AnchorTreeRow] = []
        for prompt in prompts:
            for anchor in self.select_top_anchors(prompt, anchors_per_prompt):
                rows.append(self.build_tree_for_anchor(prompt, anchor))
        return rows

    def write_jsonl(self, path: str | Path, rows: Iterable[AnchorTreeRow]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "\n".join(row.to_json() for row in rows) + "\n",
            encoding="utf-8",
        )

    def _root_children(self, observed_token_id: int, probs: Sequence[float]) -> list[int]:
        if self.tree_spec.root_width < 1:
            raise ValueError("root_width must be at least 1.")
        root_children = [observed_token_id]
        alternatives = topk_excluding(
            probs,
            excluded_ids={observed_token_id},
            k=self.tree_spec.root_width - 1,
        )
        root_children.extend(token_id for token_id, _ in alternatives)
        return root_children

    def _append_depth_two_nodes(
        self,
        *,
        prompt_prefix: list[int],
        anchor_token_index: int,
        nodes: list[TreeNode],
        edges: list[tuple[int, int]],
    ) -> None:
        frontier_contexts: list[list[int]] = []
        frontier_parents: list[int] = []
        for local_rank in self.tree_spec.expanded_root_ranks:
            parent_id = local_rank + 1
            frontier_parents.append(parent_id)
            frontier_contexts.append(prompt_prefix + [nodes[parent_id].token_id])

        child_logits = self.model.next_logits_batch(frontier_contexts)
        for parent_id, logits in zip(frontier_parents, child_logits, strict=True):
            self._append_children_from_logits(
                parent_id=parent_id,
                logits=logits,
                anchor_token_index=anchor_token_index,
                nodes=nodes,
                edges=edges,
            )

    def _append_children_from_logits(
        self,
        *,
        parent_id: int,
        logits: Sequence[float],
        anchor_token_index: int,
        nodes: list[TreeNode],
        edges: list[tuple[int, int]],
    ) -> None:
        probs = _softmax(logits)
        picks = topk_excluding(probs, excluded_ids=set(), k=self.tree_spec.child_width)
        for local_child_rank, (child_token_id, _global_rank) in enumerate(picks):
            node_id = len(nodes)
            edges.append((parent_id, node_id))
            nodes.append(
                TreeNode(
                    node_id=node_id,
                    parent_id=parent_id,
                    token_id=child_token_id,
                    token_text=self.model.decode_token(child_token_id),
                    top_k_rank=local_child_rank,
                    depth=2,
                    position_id=anchor_token_index + 2,
                )
            )
