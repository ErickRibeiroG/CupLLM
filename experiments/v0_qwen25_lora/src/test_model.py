from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


print("Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Carregando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

print("Modelo carregado com sucesso!")


question = "Quem são os personagens principais de Cuphead?"

messages = [
    {
        "role": "user",
        "content": question
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
)

print("\nGerando resposta...\n")

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("Pergunta:")
print(question)

print("\nResposta:")
print(response)