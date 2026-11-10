import json
from pathlib import Path


INPUT_FILE = Path("data/raw/cuphead_simples.json")
OUTPUT_FILE = Path("data/processed/cuphead_chat.json")


def load_dataset():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_dataset(data):
    valid_data = []
    duplicates = set()
    questions_seen = set()

    for index, item in enumerate(data):
        if "question" not in item or "answer" not in item:
            print(f"[ERRO] Item {index} não possui question ou answer.")
            continue

        question = item["question"].strip()
        answer = item["answer"].strip()

        if not question or not answer:
            print(f"[ERRO] Item {index} possui question/answer vazio.")
            continue

        if question.lower() in questions_seen:
            duplicates.add(question)

        questions_seen.add(question.lower())

        valid_data.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ]
            }
        )

    return valid_data, duplicates


def save_dataset(data):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    print("Carregando dataset...")

    data = load_dataset()

    print(f"Registros encontrados: {len(data)}")

    processed_data, duplicates = validate_dataset(data)

    print(f"Registros válidos: {len(processed_data)}")

    if duplicates:
        print("\nPerguntas duplicadas:")
        for question in duplicates:
            print(f"- {question}")
    else:
        print("Nenhuma pergunta duplicada encontrada.")

    save_dataset(processed_data)

    print(f"\nDataset processado salvo em:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()