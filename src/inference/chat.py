"""Faz uma pergunta ao modelo ajustado (ou ao base, para comparação).

Uso:
    python -m src.inference.chat "Quem são os protagonistas de Cuphead?"
    python -m src.inference.chat --model Qwen/Qwen3-1.7B "Quem é King Dice?"
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import FINETUNED_MODEL_ID, SYSTEM_PROMPT


def build_messages(question: str, system_prompt: str | None = SYSTEM_PROMPT) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": question})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--model", default=FINETUNED_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-system-prompt", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to(device).eval()

    messages = build_messages(args.question, None if args.no_system_prompt else SYSTEM_PROMPT)
    # O treino usou respostas sem raciocínio, que o template do Qwen3 renderiza com um bloco
    # <think> vazio. enable_thinking=False reproduz esse mesmo formato na inferência.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()
