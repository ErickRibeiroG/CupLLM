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
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
)

print("4. Entrada tokenizada!")
print(f"Quantidade de tokens: {inputs['input_ids'].shape[1]}")

print("5. Fazendo forward pass...")

with torch.no_grad():
    outputs = model(**inputs)

print("6. Forward pass concluído!")

print(f"Shape da saída: {outputs.logits.shape}")

print("7. Teste concluído!")