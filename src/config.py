"""Constantes compartilhadas pelo pipeline (dados, treino, inferência e avaliação)."""

from pathlib import Path

# Modelos
BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
FINETUNED_MODEL_ID = "Er1ckRg/cuphead_basico"

# Dataset no Hugging Face Hub. A revisão fixa garante que todos usem os mesmos 84 exemplos.
DATASET_REPO_ID = "Er1ckRg/cuphead_basico"
DATASET_FILENAME = "dataset.jsonl"
DATASET_REVISION = "dd714567cfd0ff14195bd04d2b05b7192f5d1b84"

# System prompt presente em todos os exemplos do dataset (copiado literalmente do dataset.jsonl).
SYSTEM_PROMPT = (
    "Você é um especialista no jogo Cuphead e sua DLC The Delicious Last Course. "
    "Responda perguntas de forma precisa e factual sobre a campanha, chefes, mecânicas e conteúdo do jogo."
)

# Diretórios locais (ignorados pelo git)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
VALIDATION_FILE = PROCESSED_DIR / "validation.jsonl"

SEED = 42
