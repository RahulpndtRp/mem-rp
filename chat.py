#!/usr/bin/env python3
import uuid

from my_mem.configs.base import MemoryConfig
from my_mem.client import MemoryClient


def main():
    # 1. Initialize client
    user_id = f"u-{uuid.uuid4().hex[:6]}"
    user_id = "u-378815"  # For reuse/testing

    mem = MemoryClient(MemoryConfig(), top_k=5)
    print(f"Your user_id is {user_id}. Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        # 2. Add message (STM + LTM update)
        mem.add_message(user_input, user_id=user_id, infer=True)

        # 3. Query RAG (using memory before it changes)
        response = mem.query_rag(user_input, user_id=user_id)

        # 4. Display result
        print("\nBot:", response["answer"])
        if response.get("sources"):
            print("Sources:")
            for src in response["sources"]:
                print(f" • {src['id']} → {src['text']}")
        print()


if __name__ == "__main__":
    main()
