import pytest

from continuation_tree import PromptRecord
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


def test_hf_dataset_to_prompts_can_mark_response_only_anchor_region(monkeypatch):
    class DummyTokenizer:
        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            mapping = {
                "Question: ": [10, 11],
                "\nAnswer: ": [12],
                "42": [13, 14],
            }
            return {"input_ids": mapping[text]}

    class DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(_name):
            return DummyTokenizer()

    class DummyTransformersModule:
        AutoTokenizer = DummyAutoTokenizer

    monkeypatch.setattr(
        "continuation_tree.hf._require_dependency",
        lambda module_name, package_name: DummyTransformersModule,
    )

    dataset = [{"id": "row-1", "prompt": "Question: ", "response": "42"}]
    prompts = hf_dataset_to_prompts(
        dataset,
        input_ids_column=None,
        text_column=None,
        prompt_column="prompt",
        response_column="response",
        tokenizer_name="dummy-tokenizer",
        text_joiner="\nAnswer: ",
    )

    assert prompts == [
        PromptRecord(
            prompt_id="row-1",
            input_ids=(10, 11, 12, 13, 14),
            anchor_start_index=3,
        )
    ]
