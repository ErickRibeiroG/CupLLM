---
base_model: Qwen/Qwen3-1.7B
datasets:
- Er1ckRg/cuphead_basico
language:
- pt
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3
- sft
- trl
- full-fine-tuning
- question-answering
- portuguese
- cuphead
---

# Qwen3-1.7B fine-tuned for Cuphead Q&A (Brazilian Portuguese)

A full-parameter supervised fine-tune of [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) that answers factual questions about the video game **Cuphead** and its DLC **The Delicious Last Course**, in Brazilian Portuguese.

- **Code:** [github.com/ErickRibeiroG/CupLLM](https://github.com/ErickRibeiroG/CupLLM)
- **Dataset:** [Er1ckRg/cuphead_basico](https://huggingface.co/datasets/Er1ckRg/cuphead_basico)

## Model description and intended use

- **Task:** single-turn, closed-domain question answering (bosses, mechanics, items, DLC, development trivia).
- **Language:** Portuguese (pt-BR).
- **Method:** full fine-tuning with TRL's `SFTTrainer`. No LoRA/PEFT: all parameters were updated, and the weights are a standalone BF16 checkpoint.
- **Intended use:** learning project and demo of an end-to-end LLM fine-tuning workflow (data preparation → SFT → evaluation against the base model → publication).
- **Out of scope:** use as an authoritative source of game facts (see [Limitations](#limitations)), multi-turn conversation, and other languages or domains.

## Dataset

[Er1ckRg/cuphead_basico](https://huggingface.co/datasets/Er1ckRg/cuphead_basico), revision `dd71456`. Each of the **84** examples is a conversation with three messages:

| role | content |
|---|---|
| `system` | Same prompt in all examples: *"Você é um especialista no jogo Cuphead e sua DLC The Delicious Last Course. Responda perguntas de forma precisa e factual sobre a campanha, chefes, mecânicas e conteúdo do jogo."* |
| `user` | A question (84 unique questions) |
| `assistant` | The answer (median 82 words, max 99) |

- **How the data was created:** [PREENCHER: how the questions/answers were written, e.g. manually or LLM-generated and reviewed]
- **Train/validation split used for this model:** [PREENCHER: e.g. 76/8 with seed 42, or all 84 for training]

## Training procedure

Supervised fine-tuning with `trl.SFTTrainer`. The hyperparameters below come from the `training_args.bin` in this repository.

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Learning rate / scheduler | 1.5e-5, cosine, 10 warmup steps |
| Per-device batch size (train / eval) | 4 / 4 |
| Gradient accumulation | 1 |
| Max sequence length | 1024 tokens |
| Optimizer | AdamW 8-bit (`adamw_bnb_8bit`), weight decay 0.01, max grad norm 1.0 |
| Precision | bf16, gradient checkpointing |
| Loss | assistant tokens only (`assistant_only_loss=True`), no packing |
| Evaluation / logging | every 50 steps |
| Seed | 42 |

- **Hardware and training time:** [PREENCHER: GPU, e.g. Kaggle T4/P100, and duration]
- **Final training / validation loss:** [PREENCHER]

**Chat template and thinking mode.** For assistant-only loss, TRL 1.7.0 swaps Qwen3's template for a training variant (`qwen3_training.jinja`) that always renders a `<think>…</think>` block. The block is empty here, because the dataset has no reasoning traces, and it falls inside the loss mask. The model therefore learned to answer without reasoning. **Use `enable_thinking=False` at inference**, which produces the same empty block in the prompt.

**Framework versions:** TRL 1.7.0, Transformers 5.12.1, PyTorch 2.10.0+cu128, Datasets 4.8.5, Tokenizers 0.22.2.

## Evaluation

Base model vs. fine-tuned model on **34 held-out questions** written for this evaluation: 29 about Cuphead and 5 general-knowledge questions to check for regressions.

**Contamination check.** An automatic check compares every test question with all 84 dataset questions after normalization. It looks for exact matches, character similarity ≥ 0.80 and content-word Jaccard ≥ 0.60. No pair crossed a threshold. The highest values were 0.795 and 0.50, from questions that share a template ("Como funciona a arma X?") but not a subject.

**Setup.** Both models get the same system prompt (the one from training), `enable_thinking=False`, greedy decoding (`do_sample=False`), `max_new_tokens=512` and the same dtype.

**Metric: key-fact coverage.** Each question has 1–4 required facts with accepted variants, such as `rato|mouse` for "What animal is Werner Werman?". An answer's score is the fraction of facts it contains, matched after lowercasing and accent stripping. This fits factual entity Q&A and is deterministic and auditable. Overlap metrics like ROUGE would mostly reward imitating the dataset's style. Each in-domain question is also labeled by whether its facts appear in training examples on the same subject: *covered* (14), *partial* (5) or *new* (10).

### Results

[PREENCHER: paste the tables from `eval_summary.md`]

| Subset | Model | n | Fact coverage (%) | Fully correct (%) | Mean tokens |
|---|---|---|---|---|---|
| Cuphead (in-domain) | Qwen3-1.7B (base) | 29 | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| Cuphead (in-domain) | cuphead_basico | 29 | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| General knowledge | Qwen3-1.7B (base) | 5 | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| General knowledge | cuphead_basico | 5 | [PREENCHER] | [PREENCHER] | [PREENCHER] |

[PREENCHER: by-coverage table and paired comparison, plus 2–3 sentences interpreting them]

Per-question answers and scores: [`eval/`](https://github.com/ErickRibeiroG/CupLLM/tree/main/eval) in the GitHub repository.

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Er1ckRg/cuphead_basico"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

system_prompt = (
    "Você é um especialista no jogo Cuphead e sua DLC The Delicious Last Course. "
    "Responda perguntas de forma precisa e factual sobre a campanha, chefes, mecânicas e conteúdo do jogo."
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Que animal é Werner Werman?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,  # matches the training format (see "Training procedure")
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Example output: [PREENCHER: paste the model's real answer to this question from `eval_responses.csv`]

## Limitations

- **Tiny dataset.** With 84 examples, the model mostly learns the answer *style* and a limited set of facts. Coverage of the game is partial.
- **Training data may contain factual errors.** [CONFIRMAR antes de publicar] The dataset was not systematically fact-checked. Spot checks found statements that appear to be wrong, such as a boss "Dr. Finkelstein" listed on Inkwell Isle III and Grim Matchstick described as an airplane fight. A fine-tuned model can reproduce these errors with confidence.
- **Hallucination.** Like its base model, it can produce fluent but false details. The key-fact metric does not penalize extra wrong content.
- **Thinking mode.** Trained without reasoning traces. Behaviour with `enable_thinking=True` was not evaluated.
- **Small evaluation set.** 34 questions; differences of a few points are not statistically meaningful. The reference facts were written for this project [CONFIRMAR: and reviewed by the author].
- **Language.** Trained and evaluated in Portuguese only.

## License

Released under Apache-2.0, the same license as the base model [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). The dataset is MIT-licensed. Cuphead is a trademark of StudioMDHR. This project is unofficial and not affiliated with StudioMDHR.
