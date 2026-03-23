import json
from pathlib import Path

from continuation_tree import ContinuationTreeGenerator, MockCausalLM, PromptRecord, TreeSpec


def _build_model() -> MockCausalLM:
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
    return MockCausalLM(vocab=vocab, prefill_map=prefill, next_map=next_map)


def test_anchor_scoring_prefers_branchy_position():
    prompt = PromptRecord(prompt_id="p1", input_ids=(1, 2, 3, 4))
    generator = ContinuationTreeGenerator(_build_model())
    anchors = generator.select_top_anchors(prompt, 2)
    assert [anchor.token_index for anchor in anchors] == [2, 1]
    assert anchors[0].observed_token_id == 4


def test_anchor_scoring_can_skip_prompt_prefix():
    prompt = PromptRecord(prompt_id="p1", input_ids=(1, 2, 3, 4), anchor_start_index=2)
    generator = ContinuationTreeGenerator(_build_model())
    anchors = generator.select_top_anchors(prompt, 2)
    assert [anchor.token_index for anchor in anchors] == [2]
    assert anchors[0].observed_token_id == 4


def test_forced_observed_token_becomes_root_rank_zero():
    prompt = PromptRecord(prompt_id="p1", input_ids=(1, 2, 3, 4))
    generator = ContinuationTreeGenerator(_build_model())
    anchor = generator.select_top_anchors(prompt, 1)[0]
    row = generator.build_tree_for_anchor(prompt, anchor)
    assert row.node_token_ids[1:4] == (4, 5, 6)
    assert row.top_K[1:4] == (0, 1, 2)
    assert row.node_text[1:4] == ("across", "over", "a")


def test_tree_serialization_matches_fixed_shape():
    prompt = PromptRecord(prompt_id="p1", input_ids=(1, 2, 3, 4))
    generator = ContinuationTreeGenerator(_build_model(), tree_spec=TreeSpec())
    anchor = generator.select_top_anchors(prompt, 1)[0]
    row = generator.build_tree_for_anchor(prompt, anchor)
    assert row.edges == ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (2, 7))
    assert row.top_K == (None, 0, 1, 2, 0, 1, 0, 1)
    payload = json.loads(row.to_json())
    assert payload["anchor_token_index"] == 2
    assert payload["node_position_ids"] == [2, 3, 3, 3, 4, 4, 4, 4]


def test_packed_and_naive_expansion_match():
    prompt = PromptRecord(prompt_id="p1", input_ids=(1, 2, 3, 4))
    generator = ContinuationTreeGenerator(_build_model())
    anchor = generator.select_top_anchors(prompt, 1)[0]
    packed = generator.build_tree_for_anchor(prompt, anchor)
    naive = generator.build_tree_for_anchor_naive(prompt, anchor)
    assert packed == naive


def test_jsonl_writer_emits_rows(tmp_path: Path):
    prompt = PromptRecord(prompt_id="p1", input_ids=(1, 2, 3, 4))
    generator = ContinuationTreeGenerator(_build_model())
    rows = generator.build_rows([prompt], anchors_per_prompt=1)
    destination = tmp_path / "trees.jsonl"
    generator.write_jsonl(destination, rows)
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["prompt_id"] == "p1"
