#!/usr/bin/env python3
import uuid
from my_mem.configs.base import MemoryConfig
from my_mem.memory.main import Memory
from my_mem.rag.rag_pipeline import RAGPipeline  # adjust import if your module path differs

def main():
    # 1) Bootstrap
    config  = MemoryConfig()
    mem     = Memory(config)
    rag     = RAGPipeline(mem, top_k=3)
    user_id = f"u-{uuid.uuid4().hex[:6]}"
    user_id = 'u-378815'

    print(f"Your user_id is {user_id}. Type 'exit' or 'quit' to stop.\n")

    while True:
        user = input("You: ")
        if user.strip().lower() in ("exit", "quit"):
            break

        # 2) Embed the user turn and add to short-term memory
        vec = mem.embedder.embed(user, "add")
        mem.short_term.add(user_id=user_id, text=user, embedding=vec)

        # 3) Also extract & persist any facts into long-term memory
        mem.add(user, user_id=user_id, infer=True)

        # 4) Fire the RAG pipeline
        out = rag.query(user, user_id=user_id)
        print("\nBot:", out["answer"])
        if out.get("sources"):
            print("Sources:")
            for src in out["sources"]:
                print(f" • {src['id']} → {src['text']}")
        print()

if __name__ == "__main__":
    main()
