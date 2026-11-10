"""Fine-tuning supervisionado (SFT) completo do Qwen3-1.7B no dataset de perguntas sobre Cuphead.

Reconstrução do treino que gerou Er1ckRg/cuphead_basico (o notebook original roda no Kaggle).
Os hiperparâmetros vêm de configs/sft_full_qwen3_1.7b.yaml e podem ser sobrescritos na linha de comando.

Uso (a partir da raiz do repositório, com GPU):
    python -m src.data.prepare_dataset
    python -m src.training.train --config configs/sft_full_qwen3_1.7b.yaml

Teste rápido em CPU (modelo minúsculo, 2 passos):
    python -m src.training.train --config configs/sft_full_qwen3_1.7b.yaml \
        --model_name_or_path trl-internal-testing/tiny-Qwen3ForCausalLM --max_steps 2 \
        --optim adamw_torch --bf16 false --output_dir runs/smoke

Publicar no Hub exige --push_to_hub true e a variável de ambiente HF_TOKEN (nunca token no código).
"""

import os
from dataclasses import dataclass, field

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, TrlParser

from src.config import BASE_MODEL_ID, TRAIN_FILE, VALIDATION_FILE


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(default=BASE_MODEL_ID)
    train_file: str = field(default=str(TRAIN_FILE))
    validation_file: str = field(default=str(VALIDATION_FILE))


def load_splits(script_args: ScriptArguments):
    data_files = {"train": script_args.train_file}
    if os.path.exists(script_args.validation_file):
        data_files["validation"] = script_args.validation_file

    missing = [path for path in data_files.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"{missing} não encontrado. Rode antes: python -m src.data.prepare_dataset")

    return load_dataset("json", data_files=data_files)


def main() -> None:
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()

    if training_args.push_to_hub and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("push_to_hub=true exige a variável de ambiente HF_TOKEN.")

    dataset = load_splits(script_args)
    eval_dataset = dataset.get("validation")
    if eval_dataset is None and training_args.eval_strategy != "no":
        print("Sem arquivo de validação: desativando avaliação durante o treino.")
        training_args.eval_strategy = "no"

    print(dataset)

    # Passar o id como string faz o SFTTrainer carregar modelo e tokenizer (dtype do checkpoint, bf16).
    # Sem peft_config: todos os parâmetros são treinados (fine-tuning completo).
    trainer = SFTTrainer(
        model=script_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"Parâmetros treináveis: {trainable:,} de {total:,} ({trainable / total:.0%})")

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    trainer.save_state()  # grava trainer_state.json com o histórico de loss
    trainer.save_model(training_args.output_dir)  # com push_to_hub=true, também envia ao Hub
    print(f"Modelo salvo em {training_args.output_dir}")


if __name__ == "__main__":
    main()
