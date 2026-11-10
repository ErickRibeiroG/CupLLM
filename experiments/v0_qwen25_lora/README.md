# v0: primeiro protótipo (Qwen2.5-0.5B-Instruct + LoRA)

Primeira tentativa do projeto, **não** é o modelo publicado em `Er1ckRg/cuphead_basico`.
Está aqui só como histórico.

| | v0 (esta pasta) | Modelo publicado (`src/`) |
|---|---|---|
| Base | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen3-1.7B` |
| Método | LoRA r=8 (q/k/v/o), `transformers.Trainer` | Fine-tuning completo, `trl.SFTTrainer` |
| Dados | 72 pares pergunta/resposta locais (58 treino / 14 validação) | 84 conversas do dataset no Hub |

- O treino desta versão nunca gerou um modelo salvo neste repositório.
- Os scripts usam caminhos do layout antigo (`data/raw`, `data/processed`). Para executá-los, ajuste as constantes no topo de cada arquivo.
