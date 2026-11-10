import pytest

from src.config import SYSTEM_PROMPT
from src.data.prepare_dataset import split_examples, validate_examples


def make_example(question: str, answer: str = "resposta", system: str = SYSTEM_PROMPT) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def test_validate_keeps_well_formed_examples():
    examples = [make_example("Pergunta 1?"), make_example("Pergunta 2?")]
    assert validate_examples(examples) == examples


@pytest.mark.parametrize(
    "bad",
    [
        {"question": "formato antigo", "answer": "x"},
        {"messages": [{"role": "user", "content": "sem system"}, {"role": "assistant", "content": "x"}]},
        make_example("Pergunta?", answer="   "),
    ],
)
def test_validate_drops_malformed_examples(bad):
    assert validate_examples([bad]) == []


def test_validate_drops_case_insensitive_duplicates():
    examples = [make_example("Quem é King Dice?"), make_example("quem é king dice?  ")]
    assert len(validate_examples(examples)) == 1


def test_split_is_deterministic_disjoint_and_complete():
    examples = [make_example(f"Pergunta {i}?") for i in range(84)]
    train_a, val_a = split_examples(examples, val_size=0.1, seed=42)
    train_b, val_b = split_examples(examples, val_size=0.1, seed=42)

    assert (train_a, val_a) == (train_b, val_b)
    assert len(val_a) == 8 and len(train_a) == 76
    questions = lambda xs: {x["messages"][1]["content"] for x in xs}  # noqa: E731
    assert questions(train_a).isdisjoint(questions(val_a))
    assert questions(train_a) | questions(val_a) == questions(examples)


def test_split_without_validation():
    examples = [make_example(f"Pergunta {i}?") for i in range(10)]
    train, val = split_examples(examples, val_size=0)
    assert len(train) == 10 and val == []
