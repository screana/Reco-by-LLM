from ollama import chat

response = chat(
    model = "ministral-3",
    messages = [
        {"role":"user", "content":"こんにちは"}
    ]
)

print(response)
print("--"*10)
print(response["message"]["content"])