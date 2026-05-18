from datasets import load_dataset

# ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

SYSTEM_PROMPT = """
# PLACEHOLDER: system prompt goes here
"""


def call_model(messages: list[dict]) -> str:
    # PLACEHOLDER: call your model API here and return the assistant reply as a string
    raise NotImplementedError("Replace this with your model call")


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
