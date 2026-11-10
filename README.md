# CupLLM: fine-tuning do Qwen3-1.7B para perguntas sobre Cuphead

Fine-tuning supervisionado (SFT) completo do [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) com [TRL](https://github.com/huggingface/trl) para responder, em português, perguntas factuais sobre o jogo **Cuphead** e a DLC **The Delicious Last Course**. O projeto inclui uma avaliação contra o modelo base.

| | Link |
|---|---|
| Modelo | [Er1ckRg/cuphead_basico](https://huggingface.co/Er1ckRg/cuphead_basico) |
| Dataset | [Er1ckRg/cuphead_basico](https://huggingface.co/datasets/Er1ckRg/cuphead_basico) |

## Problema

Um LLM pequeno de uso geral conhece pouco de um domínio de nicho e tende a responder de forma genérica ou inventada. A pergunta do projeto: **com poucos exemplos (84) e uma GPU gratuita do Kaggle, quanto um SFT melhora um modelo de 1,7B parâmetros num domínio fechado, e quanto ele perde fora desse domínio?**

## Abordagem

1. **Dados.** 84 conversas `system / user / assistant` em pt-BR, publicadas no Hub. O script de preparo fixa a revisão do dataset, valida o formato, remove duplicatas e gera os splits de treino e validação.
2. **Treino.** Fine-tuning **completo** (sem LoRA) com `SFTTrainer`:
   - `assistant_only_loss=True`: a loss só considera a resposta, não o prompt.
   - Otimizador AdamW 8-bit, bf16 e gradient checkpointing, que reduzem o uso de memória.
   - Os hiperparâmetros estão em [`configs/sft_full_qwen3_1.7b.yaml`](configs/sft_full_qwen3_1.7b.yaml).
3. **Modo de raciocínio do Qwen3.** Com `assistant_only_loss`, o TRL usa um template de treino que sempre inclui um bloco `<think>` (vazio aqui) dentro da loss. O modelo aprendeu a responder sem raciocinar, então a inferência e a avaliação usam `enable_thinking=False`.
4. **Avaliação.** São 34 perguntas novas: 29 sobre Cuphead e 5 gerais, para detectar regressão.
   - Uma checagem automática garante que nenhuma delas se sobrepõe ao dataset.
   - Base e ajustado geram com os mesmos parâmetros e decodificação gulosa.
   - A métrica é a **cobertura de fatos-chave**: a fração dos fatos esperados que aparecem na resposta. ROUGE/BLEU não serviriam, porque premiariam imitar o estilo do dataset em vez de acertar o fato.
   - Cada pergunta também é marcada como *coberta*, *parcial* ou *nova* em relação ao treino. Isso separa "decorou o dataset" de "generalizou".

## Resultados

[PREENCHER: colar as tabelas de `eval_summary.md` depois de rodar o notebook no Kaggle]

| Subconjunto | Modelo | n | Cobertura de fatos (%) | Respostas completas (%) |
|---|---|---|---|---|
| Cuphead | Qwen3-1.7B (base) | 29 | [PREENCHER] | [PREENCHER] |
| Cuphead | cuphead_basico | 29 | [PREENCHER] | [PREENCHER] |
| Conhecimento geral | Qwen3-1.7B (base) | 5 | [PREENCHER] | [PREENCHER] |
| Conhecimento geral | cuphead_basico | 5 | [PREENCHER] | [PREENCHER] |

[PREENCHER: 2 ou 3 frases de interpretação, por exemplo o ganho em perguntas cobertas × novas]

### Limitações conhecidas

- **Dataset pequeno e não checado.** [CONFIRMAR] O dataset não passou por checagem factual sistemática. Uma leitura encontrou afirmações suspeitas, como um chefe "Dr. Finkelstein" listado na Inkwell Isle III. O modelo pode reproduzir esses erros.
- **Métrica parcial.** A métrica não pune informação extra errada, e 34 perguntas não bastam para diferenças pequenas serem significativas.

## Como reproduzir

```bash
pip install -r requirements.txt        # instale antes o torch com CUDA adequado à sua GPU

# 1. Dados: baixa do Hub e gera data/processed/{train,validation}.jsonl
python -m src.data.prepare_dataset --val-size 0.1

# 2. Treino (GPU com suporte a bf16; o modelo é salvo em runs/)
python -m src.training.train --config configs/sft_full_qwen3_1.7b.yaml

# 3. Inferência
python -m src.inference.chat "Que animal é Werner Werman?"

# Testes
pytest
```

- **Avaliação:** abra [`eval/eval_base_vs_finetuned.ipynb`](eval/eval_base_vs_finetuned.ipynb) no Kaggle (GPU, Internet ligada) e use *Run All*. O notebook gera as respostas lado a lado (CSV), as notas e as tabelas em Markdown.
- **Publicação no Hub:** `--push_to_hub true` com a variável `HF_TOKEN` definida (ver [`.env.example`](.env.example)); nenhum token fica no código. No Kaggle, use *Add-ons → Secrets*:

  ```python
  import os
  from kaggle_secrets import UserSecretsClient
  os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
  ```

  Atenção: o push pelo trainer **sobrescreve o model card** com o template do TRL. Depois dele, publique de novo o [`model_card/README.md`](model_card/README.md).
- **Split original:** o split exato usado no treino publicado não foi registrado [PREENCHER]. O padrão do script (76/8, seed 42) pode diferir.
- **Teste rápido em CPU:** troque o modelo por um Qwen3 minúsculo:

  ```bash
  python -m src.training.train --config configs/sft_full_qwen3_1.7b.yaml \
      --model_name_or_path trl-internal-testing/tiny-Qwen3ForCausalLM \
      --max_steps 2 --optim adamw_torch --bf16 false --output_dir runs/smoke
  ```

## Estrutura

```
configs/            hiperparâmetros do treino (YAML)
src/config.py       ids de modelo/dataset, revisão do dataset, system prompt
src/data/           download, validação e split do dataset
src/training/       treino com SFTTrainer
src/inference/      perguntas ao modelo pela linha de comando
eval/               prompts de teste com gabarito e notebook de avaliação (Kaggle)
model_card/         model card publicado no Hugging Face
tests/              testes do preparo de dados
experiments/v0_*    primeiro protótipo (Qwen2.5-0.5B + LoRA), mantido como histórico
```

## Stack

Python · PyTorch · Hugging Face Transformers · TRL (SFTTrainer) · Datasets · bitsandbytes · Kaggle (GPU)
