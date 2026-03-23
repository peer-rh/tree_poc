import pytest

from continuation_tree.hf import hf_dataset_to_prompts


def test_hf_dataset_to_prompts_uses_input_ids_without_external_deps():
    dataset = [
        {"id": "a", "input_ids": [1, 2, 3]},
        {"id": "b", "input_ids": [4, 5]},
    ]
    prompts = hf_dataset_to_prompts(dataset, max_examples=1)
    assert len(prompts) == 1
    assert prompts[0].prompt_id == "a"
    assert prompts[0].input_ids == (1, 2, 3)


def test_hf_dataset_to_prompts_requires_tokenizer_for_text_only():
    dataset = [{"id": "a", "text": "the fox"}]
    with pytest.raises(RuntimeError):
        hf_dataset_to_prompts(
            dataset,
            input_ids_column=None,
            text_column="text",
            tokenizer_name="dummy-tokenizer",
        )
