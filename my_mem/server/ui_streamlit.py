"""
Streamlit chat demo — run with:
    streamlit run my_mem/ui_streamlit.py
"""

import streamlit as st
from uuid import uuid4

from my_mem.configs.base import MemoryConfig
from my_mem.memory.main  import Memory
from my_mem.rag.rag_pipeline import RAGPipeline

# --- singleton objects ----------------------------------------------------
if "mem" not in st.session_state:
    st.session_state.mem = Memory(MemoryConfig())
    st.session_state.rag = RAGPipeline(st.session_state.mem)

if "user_id" not in st.session_state:
    st.session_state.user_id = f"u-{uuid4().hex[:6]}"

st.title("🧠 my_mem chat (+ citations)")

# ------------ sidebar -----------------------------------------------------
with st.sidebar:
    st.caption("**Debug**")
    if st.button("Reset store"):
        st.session_state.mem.reset()
        st.success("Memory cleared!")

# ------------ main chat ---------------------------------------------------
query = st.chat_input("Ask me something…")
if query:
    # store query in STM so it’s searchable later
    st.session_state.mem.short_term.add(query)

    rag_out = st.session_state.rag.query(query, user_id=st.session_state.user_id)
    answer  = rag_out["answer"]
    sources = rag_out["sources"]

    st.markdown(f"**Answer**  \n{answer}")

    with st.expander("Sources"):
        for s in sources:
            st.write(f"*{s['id']}*: {s['text']}")

# show STM contents (tiny console)
if st.toggle("Show short-term memory", False):
    st.json(st.session_state.mem.short_term.dump(), expanded=False)
