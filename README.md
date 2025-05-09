# 🧠 memrp – Modular Memory + RAG System

## 📌 Project Overview

`memrp` is a lightweight but production-grade memory system designed to:

* Store and retrieve **semantic + episodic** memories (vector-based)
* Maintain **short-term memory** (last N user messages)
* Perform **fact extraction** + **reconciliation** using LLMs
* Enable **source-aware** Retrieval-Augmented Generation (RAG)
* Offer a **Streamlit-based UI** with user persona selection

## 🌐 Live App

Try the demo at: [https://memlite.streamlit.app/](https://memlite.streamlit.app/)

Switch between personas using the dropdown menu in the sidebar to simulate memory contexts for different users.

---

## 🔧 Features

| Component       | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| Memory          | Supports both long-term (via FAISS) and short-term memory              |
| Fact Extraction | Uses LLM to extract structured facts from user messages                |
| Update Engine   | Applies ADD / UPDATE / DELETE / NONE logic on top of existing memories |
| Vector Store    | Default: FAISS (can be extended to Milvus, Qdrant, Pinecone, etc.)     |
| LLM Support     | OpenAI, Azure OpenAI, Ollama, LM Studio, etc.                          |
| RAG             | Retrieves context-aware memory for user questions                      |
| Streamlit UI    | Chat interface with real-time memory updates and persona switcher      |

---

## 📁 Directory Structure

```
memrp/
├── my_mem/
│   ├── memory/              # Core memory class (long + short term)
│   ├── vector_stores/       # FAISS-compatible vector DB backend
│   ├── utils/               # Fact parsing, prompts, factories
│   └── configs/             # Pydantic model-based configs
├── app.py                   # Streamlit UI frontend
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone and Install

```bash
git clone https://github.com/yourname/memrp.git
cd memrp
poetry install  # or: pip install -r requirements.txt
```

### 2. Run Locally

```bash
streamlit run app_4.py
```

### 3. Configure `.env`

```dotenv
OPENAI_API_KEY=sk-...
```

You can also switch providers (Azure, Ollama, etc.) via `configs/base.py`.

---

## 🧪 Example Usage

```python
from my_mem.configs.base import MemoryConfig
from my_mem.memory.main import Memory

mem = Memory(MemoryConfig())
mem.add("Favourite food is sushi", user_id="u1")
result = mem.search("What food do I like?", user_id="u1")
print(result)
```

---

## 🧠 Memory Architecture

* ✅ Long-term memory: Stored in FAISS with hashed vector payloads
* ✅ Short-term memory: Last N entries stored in memory ring buffer
* ✅ SQLite used for historical logging and rollback

---

## 🔍 RAG with Source Tracing

Query → Embed → Retrieve top-K memory chunks → Prompt LLM with full context → Answer with source IDs

Output example:

```
Your favorite food is sushi [1][2].

Sources:
• 2324...: Favourite food is sushi
• stm-0...: Favourite food is sushi
```

---

## 🧑‍💼 Persona Support (in UI)

In the Streamlit sidebar, you can:

* Select a persona (`u1`, `u2`, etc.)
* Maintain isolated memory tracks per persona

---

## 📜 License

[MIT](LICENSE)

---

## 🙌 Acknowledgements

Inspired by:

* [mem0](https://github.com/lepta-ai/mem0)
* [letta](https://github.com/letta-ai/letta)

---

Contributions welcome. Let’s build better memory-first agents!
