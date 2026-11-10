"""Baixa o dataset do Hub, valida o formato conversacional e gera os splits de treino/validação.

Uso (a partir da raiz do repositório):
    python -m src.data.prepare_dataset                 # validação de 10% (padrão)
    python -m src.data.prepare_dataset --val-size 0    # sem validação: treina nos 84 exemplos

Atenção: o split exato usado no treino publicado (notebook no Kaggle) não foi registrado.
O valor padrão aqui é uma escolha deste script, não o valor original.
"""

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.config import (
    DATASET_FILENAME,
    DATASET_REPO_ID,
    DATASET_REVISION,
    SEED,
    TRAIN_FILE,
    VALIDATION_FILE,
)

EXPECTED_ROLES = ["system", "user", "assistant"]


def download_dataset(revision: str = DATASET_REVISION) -> list[dict]:
    path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=DATASET_FILENAME,
        repo_type="dataset",
        revision=revision,
    )
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def validate_examples(examples: list[dict]) -> list[dict]:
    """Mantém só exemplos system/user/assistant não vazios e sem pergunta duplicada."""
    valid, seen_questions = [], set()

    for index, example in enumerate(examples):
        messages = example.get("messages")
        if not isinstance(messages, list):
            print(f"[descartado] exemplo {index}: sem campo 'messages'")
            continue

        roles = [message.get("role") for message in messages]
        if roles != EXPECTED_ROLES:
            print(f"[descartado] exemplo {index}: roles {roles}, esperado {EXPECTED_ROLES}")
            continue

        if any(not str(message.get("content", "")).strip() for message in messages):
            print(f"[descartado] exemplo {index}: mensagem vazia")
            continue

        question = messages[1]["content"].strip().lower()
        if question in seen_questions:
            print(f"[descartado] exemplo {index}: pergunta duplicada")
            continue

        seen_questions.add(question)
        valid.append({"messages": messages})

    return valid


def split_examples(examples: list[dict], val_size: float, seed: int = SEED):
    if not 0 <= val_size < 1:
        raise ValueError("--val-size deve estar em [0, 1)")

    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_size)
    return shuffled[n_val:], shuffled[:n_val]


def save_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-size", type=float, default=0.1, help="fração para validação (0 = sem validação)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--revision", default=DATASET_REVISION, help="commit do dataset no Hub")
    args = parser.parse_args()

    raw = download_dataset(args.revision)
    examples = validate_examples(raw)
    print(f"Exemplos baixados: {len(raw)} | válidos: {len(examples)}")

    train, validation = split_examples(examples, args.val_size, args.seed)
    save_jsonl(train, TRAIN_FILE)
    print(f"Treino: {len(train)} -> {TRAIN_FILE}")

    if validation:
        save_jsonl(validation, VALIDATION_FILE)
        print(f"Validação: {len(validation)} -> {VALIDATION_FILE}")
    elif VALIDATION_FILE.exists():
        VALIDATION_FILE.unlink()  # evita que o treino use um split antigo


if __name__ == "__main__":
    main()
