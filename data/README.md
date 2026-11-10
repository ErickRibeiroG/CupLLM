# Dados

A fonte dos dados é o dataset [`Er1ckRg/cuphead_basico`](https://huggingface.co/datasets/Er1ckRg/cuphead_basico) no Hugging Face Hub
(`dataset.jsonl`, 84 conversas em português, licença MIT). O script de preparo usa a revisão `dd71456` fixada em `src/config.py`.

Cada linha tem o formato:

```json
{"messages": [
  {"role": "system", "content": "Você é um especialista no jogo Cuphead ..."},
  {"role": "user", "content": "Quem são os protagonistas de Cuphead?"},
  {"role": "assistant", "content": "Os protagonistas de Cuphead são Cuphead e Mugman, ..."}
]}
```

Para gerar `data/processed/train.jsonl` e `validation.jsonl` (ignorados pelo git):

```bash
python -m src.data.prepare_dataset --val-size 0.1
```
