import json
import random
from pathlib import Path


INPUT_FILE = Path("data/processed/cuphead_chat.json")
TRAIN_FILE = Path("data/processed/train.json")
VALIDATION_FILE = Path("data/processed/validation.json")

VALIDATION_RATIO = 0.20
SEED = 42


def load_dataset():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def split_dataset(data):
    random.seed(SEED)

    data = data.copy()
    random.shuffle(data)

    validation_size = int(len(data) * VALIDATION_RATIO)

    validation_data = data[:validation_size]
    train_data = data[validation_size:]

    return train_data, validation_data


def save_dataset(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    data = load_dataset()

    print(f"Total de exemplos: {len(data)}")

    train_data, validation_data = split_dataset(data)

    save_dataset(train_data, TRAIN_FILE)
    save_dataset(validation_data, VALIDATION_FILE)

    print(f"Treinamento: {len(train_data)} exemplos")
    print(f"Validação: {len(validation_data)} exemplos")

    print("\nArquivos criados:")
    print(f"- {TRAIN_FILE}")
    print(f"- {VALIDATION_FILE}")


if __name__ == "__main__":
    main()