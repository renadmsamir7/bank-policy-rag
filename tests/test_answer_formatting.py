
import pytest

from evaluate_answers import check_answer_formatting


@pytest.mark.parametrize(
    "answer",
    [
        "The statement on whichthe transaction appeared.",
        "The deadline is withinthirty days.",
        "Thisfee is QAR 50.",
    ]
)
def test_detects_merged_words(answer):
    assert check_answer_formatting(answer) is False


@pytest.mark.parametrize(
    "answer",
    [
        "The statement on which the transaction appeared.",
        "The deadline is within thirty days.",
        "This fee is QAR 50.",
    ]
)
def test_accepts_correct_spacing(answer):
    assert check_answer_formatting(answer) is True
