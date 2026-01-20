from ollama import chat


def main() -> None:
    """ministral-3 を単独で動かす疎通テスト。"""
    response = chat(
        model="ministral-3",
        messages=[
            {"role": "user", "content": "こんにちは"},
        ],
    )

    print(response)
    print("--" * 10)
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
