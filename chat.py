"""
Minimal interactive chat REPL against the local vLLM OpenAI-compatible server.

Usage:
    python3.12 chat.py                  # talks to http://localhost:8000
    BASE_URL=http://host:port python3.12 chat.py
    MODEL=Qwen/Qwen3-4B python3.12 chat.py

Commands inside the REPL:
    /reset      clear conversation history
    /system X   set/replace the system prompt to X
    /quit       exit (Ctrl-D / Ctrl-C also work)
"""
from __future__ import annotations

import os
import sys
from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B")
SYSTEM = os.environ.get("SYSTEM", "You are a helpful assistant. Be concise.")

client = OpenAI(api_key="dummy", base_url=BASE_URL)

def banner() -> None:
    print(f"connected: {BASE_URL}   model: {MODEL}")
    print("commands: /reset  /system <prompt>  /quit\n")

def main() -> None:
    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    banner()
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user:
            continue
        if user in ("/quit", "/exit"):
            return
        if user == "/reset":
            history = [history[0]]
            print("(history cleared)\n")
            continue
        if user.startswith("/system "):
            history[0] = {"role": "system", "content": user[len("/system "):].strip()}
            print("(system prompt updated)\n")
            continue

        history.append({"role": "user", "content": user})

        print("ai> ", end="", flush=True)
        assistant_text = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=history,
                stream=True,
                temperature=0.7,
                max_tokens=1024,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    assistant_text += delta
        except Exception as e:
            print(f"\n[error] {e}")
            history.pop()  # roll back the user turn so we can retry cleanly
            continue

        print("\n")
        history.append({"role": "assistant", "content": assistant_text})

if __name__ == "__main__":
    main()
