from dataclasses import replace

import pytest

from continuation_tree import ContinuationTreeGenerator, MockCausalLM, PromptRecord
from continuation_tree.visualize import load_rows, main, render_sample, select_prompt_rows


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


def _build_row():
    prompt = PromptRecord(prompt_id="p1", input_ids=(1, 2, 3, 4))
    generator = ContinuationTreeGenerator(_build_model())
    anchor = generator.select_top_anchors(prompt, 1)[0]
    return generator.build_tree_for_anchor(prompt, anchor)


def test_load_rows_round_trip_and_render(tmp_path):
    row = _build_row()
    path = tmp_path / "trees.jsonl"
    path.write_text(f"{row.to_json()}\n", encoding="utf-8")

    rows = load_rows(path)
    assert rows == [row]

    rendered = render_sample(rows)
    assert 'sample prompt_id=p1 anchors=1' in rendered
    assert 'anchor idx=2 anchor_token_id=3 observed_token_id=4' in rendered
    assert '├── "across" (token_id=4, rank=0, pos=3)' in rendered
    assert '└── "a" (token_id=6, rank=2, pos=3)' in rendered
    assert '│   ├── "the_det" (token_id=7, rank=0, pos=4)' in rendered


def test_select_prompt_rows_supports_index_and_prompt_id():
    row = _build_row()
    alt = replace(row, prompt_id="p2")
    prompt_id, selected_rows = select_prompt_rows([row, alt], sample_index=1)
    assert prompt_id == "p2"
    assert selected_rows == [alt]

    prompt_id, selected_rows = select_prompt_rows([row, alt], prompt_id="p1")
    assert prompt_id == "p1"
    assert selected_rows == [row]

    with pytest.raises(ValueError):
        select_prompt_rows([row], prompt_id="missing")

    with pytest.raises(IndexError):
        select_prompt_rows([row], sample_index=2)


def test_main_prints_selected_sample(tmp_path, capsys):
    row = _build_row()
    alt = replace(row, prompt_id="p2")
    path = tmp_path / "trees.jsonl"
    path.write_text(f"{row.to_json()}\n{alt.to_json()}\n", encoding="utf-8")

    main(["--input", str(path), "--prompt-id", "p2"])
    stdout = capsys.readouterr().out
    assert stdout.startswith("sample prompt_id=p2 anchors=1")
