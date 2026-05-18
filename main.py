import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

MODEL = "deepseek-ai/DeepSeek-V3.2"

SYSTEM_PROMPT = """
# PLACEHOLDER: system prompt goes here
"""


def call_model(messages: list[dict]) -> str:
    formatted = []
    for msg in messages:
        if msg["role"] == "user":
            formatted.append({"role": "user", "content": [{"type": "text", "text": msg["content"]}]})
        else:
            formatted.append(msg)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + formatted,
    )
    return response.choices[0].message.content


def run_agent():
    messages = []

    print("Agent ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        reply = call_model(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    run_agent()
