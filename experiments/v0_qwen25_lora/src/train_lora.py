import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = "data/processed/train.json"
VALIDATION_FILE = "data/processed/validation.json"


def load_data():
    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VALIDATION_FILE,
        },
    )

    return dataset


def format_conversation(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def tokenize(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
    )


def main():

    print("Carregando dataset...")
    dataset = load_data()

    print(dataset)

    print("\nCarregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Formatando conversas...")

    dataset = dataset.map(
        lambda example: format_conversation(
            example,
            tokenizer
        )
    )

    print("\nTokenizando dataset...")

    tokenized_dataset = dataset.map(
        lambda example: tokenize(
            example,
            tokenizer
        ),
        remove_columns=dataset["train"].column_names,
    )

    print("\nCarregando modelo...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    print("\nConfigurando LoRA...")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="models/cuphead-qwen",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    print("\nIniciando treinamento...\n")

    trainer.train()

    print("\nSalvando modelo...")

    trainer.save_model(
        "models/cuphead-qwen"
    )

    tokenizer.save_pretrained(
        "models/cuphead-qwen"
    )

    print("\nTreinamento concluído!")


if __name__ == "__main__":
    main()