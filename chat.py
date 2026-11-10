"""
A simple terminal chatbot for a local server that speaks the OpenAI API
(LM Studio, Ollama, vLLM...). Uses the same environment variables as mini_rag.py.

Install:  pip install openai

Run (PowerShell):
    $env:LLM_BASE_URL="http://localhost:PORT/v1"   # if the port isn't 1234
    $env:LLM_MODEL="model-id"
    python chat.py

Commands in the chat:  /reset  clears the history,  /exit  quits (or Ctrl+C).
"""
import os

from openai import OpenAI

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.5-9b")
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely and to the point, in the language the user writes in."
MAX_HISTORY = 20  # how many recent messages to send to the model (protects the context window)


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    history: list[dict] = []
    print(f"Model: {MODEL} | Server: {BASE_URL}")
    print("Commands: /reset, /exit\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history.clear()
            print("History cleared.\n")
            continue

        history.append({"role": "user", "content": user})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-MAX_HISTORY:]

        reply, finish = "", None
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                reasoning_effort="none",  # no "thinking": the answer starts right away
                stream=True,  # tokens appear as they're generated
            )
            print("Bot: ", end="", flush=True)
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                piece = choice.delta.content or ""
                print(piece, end="", flush=True)
                reply += piece
                if choice.finish_reason:
                    finish = choice.finish_reason
            print()
            if finish == "length":
                print("[NOTE: the answer was cut off by the token limit]")
        except KeyboardInterrupt:
            print("\n[interrupted]")  # Ctrl+C during generation keeps the partial answer
        except Exception as e:  # no connection, wrong model, etc.
            history.pop()  # remove the question that got no answer
            print(f"\nError: {e}\nCheck that the LM Studio server is running and the model is loaded.\n")
            continue

        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
