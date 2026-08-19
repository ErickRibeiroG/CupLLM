import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


print("1. Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("2. Carregando modelo...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("3. Modelo carregado!")


question = "Quem são Cuphead e Mugman?"

messages = [
    {
        "role": "user",
        "content": question,
    },
    {
        "role": "assistant",
        "content": "Cuphead e Mugman são os personagens principais do jogo Cuphead.",
    },
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
)

# Para treinamento causal, usamos os próprios tokens como labels.
inputs["labels"] = inputs["input_ids"].clone()

print("4. Dados preparados!")
print(f"Tokens: {inputs['input_ids'].shape[1]}")

print("5. Forward pass + cálculo da loss...")

outputs = model(**inputs)

loss = outputs.loss

print(f"Loss inicial: {loss.item()}")

print("6. Executando backward...")

loss.backward()

print("7. Backward concluído!")

print("8. Teste de treinamento concluído!")