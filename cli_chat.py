from __future__ import annotations

import os
import sys

import requests


def _post(base_url: str, payload: dict) -> dict:
    url = base_url.rstrip("/") + "/api/chat"
    response = requests.post(url, json=payload, timeout=60)
    try:
        data = response.json()
    except Exception:
        raise RuntimeError(f"Non-JSON response ({response.status_code}): {response.text}")

    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {data}")
    return data


def main() -> int:
    base_url = os.getenv("CHATBOT_BASE_URL", "http://localhost:5000")

    # Start new conversation (no conversation_id) -> greeting
    data = _post(base_url, {})
    conversation_id = data.get("conversation_id")
    reply = data.get("reply")
    if not conversation_id:
        print("Failed to start conversation. Response:", data)
        return 1

    print(f"Assistant: {reply}")
    print("Type /exit to quit, /new to start over.")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not user_text:
            continue

        if user_text.lower() == "/exit":
            return 0

        if user_text.lower() == "/new":
            data = _post(base_url, {})
            conversation_id = data.get("conversation_id")
            print(f"Assistant: {data.get('reply')}")
            continue

        try:
            data = _post(base_url, {"conversation_id": conversation_id, "message": user_text})
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"Assistant: {data.get('reply')}")


if __name__ == "__main__":
    sys.exit(main())
