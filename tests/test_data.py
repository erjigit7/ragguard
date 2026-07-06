from datasets import Dataset

from ragguard.data import _to_pairs


def _fake_split(rows):
    return Dataset.from_dict(
        {
            "context": [r["context"] for r in rows],
            "output": [r["output"] for r in rows],
            "hallucination_labels_processed": [r["flags"] for r in rows],
        }
    )


def test_grounded_answer_gets_label_1():
    split = _fake_split(
        [{"context": "The sky is blue.", "output": "The sky is blue.", "flags": {"evident_conflict": 0, "baseless_info": 0}}]
    )
    pairs = _to_pairs(split)
    assert pairs["label"] == [1]


def test_evident_conflict_gets_label_0():
    split = _fake_split(
        [{"context": "The sky is blue.", "output": "The sky is green.", "flags": {"evident_conflict": 1, "baseless_info": 0}}]
    )
    pairs = _to_pairs(split)
    assert pairs["label"] == [0]


def test_baseless_info_gets_label_0():
    split = _fake_split(
        [{"context": "The sky is blue.", "output": "The sky is blue and it costs $5.", "flags": {"evident_conflict": 0, "baseless_info": 1}}]
    )
    pairs = _to_pairs(split)
    assert pairs["label"] == [0]


def test_preserves_context_and_answer_text():
    split = _fake_split(
        [{"context": "ctx", "output": "ans", "flags": {"evident_conflict": 0, "baseless_info": 0}}]
    )
    pairs = _to_pairs(split)
    assert pairs["context"] == ["ctx"]
    assert pairs["answer"] == ["ans"]
