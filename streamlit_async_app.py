import streamlit as st
import uuid, os, json, asyncio

from my_mem.client import AsyncMemoryClient
from my_mem.configs.base import MemoryConfig, LlmConfig

# Setup API key
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

# Config for async LLM
config = MemoryConfig(
    llm=LlmConfig(provider="openai_async", config={})
)

DATA_DIR = "user_data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
os.makedirs(DATA_DIR, exist_ok=True)

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return []

def save_users(users: list):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

st.set_page_config(page_title="MEM-RP | Made with ❤️ + LLMs", page_icon="📘")

if "users" not in st.session_state:
    st.session_state.users = load_users()
    if not st.session_state.users:
        st.session_state.users = []
        st.session_state.selected_user = ""
if "selected_user" not in st.session_state:
    st.session_state.selected_user = st.session_state.users[0] if st.session_state.users else ""
if "mem" not in st.session_state:
    st.session_state.mem = AsyncMemoryClient(config)
if "session_chat" not in st.session_state:
    st.session_state.session_chat = []
if "last_procedural" not in st.session_state:
    st.session_state.last_procedural = []

mem: AsyncMemoryClient = st.session_state.mem

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    top_row = st.columns([5, 1])
    with top_row[0]:
        st.markdown("### 👤 Select User")
    with top_row[1]:
        if st.session_state.users and st.button("🗑️", help="Delete current user", key="delete_user_top"):
            st.session_state.users.remove(st.session_state.selected_user)
            st.session_state.selected_user = st.session_state.users[0] if st.session_state.users else ""
            st.session_state.session_chat = []
            st.session_state.last_procedural = []
            save_users(st.session_state.users)
            st.rerun()

    if st.session_state.users:
        selected_index = (
            st.session_state.users.index(st.session_state.selected_user)
            if st.session_state.selected_user in st.session_state.users else 0
        )
        user = st.selectbox("User ID", st.session_state.users, index=selected_index)
        if user != st.session_state.selected_user:
            st.session_state.last_procedural = []
        st.session_state.selected_user = user
    else:
        st.info("No users available. Add a new user to begin.")

    new_user_label = st.text_input("Optional Agent Name", value="")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Add User"):
            new_user = new_user_label.strip() or f"User-{uuid.uuid4().hex[:4]}"
            st.session_state.users.append(new_user)
            st.session_state.selected_user = new_user
            save_users(st.session_state.users)
            st.session_state.session_chat = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All Users"):
            st.session_state.users = []
            st.session_state.selected_user = ""
            st.session_state.session_chat = []
            st.session_state.last_procedural = []
            save_users([])
            st.success("✅ All users cleared.")
            st.rerun()

    with st.expander("⚙️ Agent Settings", expanded=False):
        stream_enabled = st.toggle("Enable Streaming", value=True)
    

    st.markdown("---")
    if st.button("📝 Summarize Chat to Procedural Memory"):
        if st.session_state.session_chat:
            with st.spinner("📚 Summarizing chat..."):
                chat_history = [
                    {"role": "user", "content": entry["user"]} if "user" in entry else {"role": "assistant", "content": entry["bot"]}
                    for entry in st.session_state.session_chat
                ]
                result = asyncio.run(mem.summarize_procedural(chat_history, user_id=st.session_state.selected_user))
                st.success("✅ Procedural memory saved.")
        else:
            st.warning("No chat history available for summarization.")

    if st.button("📖 Show Procedural Memories") and st.session_state.selected_user:
        with st.spinner("Retrieving procedural memories..."):
            all_memories = asyncio.run(mem.get_all_memories(user_id=st.session_state.selected_user))
            procedural = [
                mem for mem in all_memories.get("results", [])
                if mem.get("metadata", {}).get("memory_type") == "procedural"
            ]
            st.session_state.last_procedural = procedural
            st.session_state.show_summary_now = True
            if procedural:
                st.toast(f"✅ {len(procedural)} procedural memories loaded.")
            else:
                st.info("No procedural memories found.")

    # Add missing Clear All Memories button here
    if st.button("🧹 Clear All Memories"):
        if st.session_state.selected_user:
            asyncio.run(mem.delete_all_memories(user_id=st.session_state.selected_user))
            st.success("✅ Cleared all memories for the selected user.")

# -----------------------------
# Chat Display
# -----------------------------
st.markdown("# 💬 MeM-RP Chat")

if not st.session_state.users:
    st.info("No users available. Please add a user to begin chatting.")
    st.stop()

if st.session_state.get("last_procedural"):
    expanded_state = st.session_state.pop("show_summary_now", False)
    with st.expander("📘 View Procedural Summaries", expanded=expanded_state):
        st.markdown("<div style='max-height: 300px; overflow-y: auto;'>", unsafe_allow_html=True)
        for i, p in enumerate(st.session_state.last_procedural, 1):
            st.markdown(f"**Summary #{i}**")
            st.markdown(f'''```markdown
{p['memory']}
```''', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("❌ Clear Displayed Summaries"):
            st.session_state.last_procedural = []
            st.rerun()

chat_container = st.container()
st.markdown("""
<style>
    section.main > div { max-width: 100% !important; }
</style>
""", unsafe_allow_html=True)

with chat_container:
    for msg in st.session_state.session_chat:
        with st.chat_message("user"):
            st.markdown(msg["user"])
        with st.chat_message("assistant"):
            st.markdown(msg.get("bot", "_Thinking..._"))

# -----------------------------
# Input Handler
# -----------------------------
user_input = st.chat_input("Type your message here...")

async def handle_input(user_input: str):
    # Clear procedural summaries when a new message is sent
    st.session_state.last_procedural = []
    user_id = st.session_state.selected_user
    st.session_state.session_chat.append({"user": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if stream_enabled:
            with st.spinner("🧠 Thinking (streaming)..."):
                response_container = st.empty()
                full_response = ""
                async for token in mem.stream_rag(user_input, user_id=user_id):
                    full_response += token
                    response_container.markdown(full_response + "▌")
                response_container.markdown(full_response.strip())
                st.session_state.session_chat[-1]["bot"] = full_response.strip()
        else:
            with st.spinner("🧠 Thinking..."):
                result = await mem.query_rag(user_input, user_id=user_id)
                reply = result["answer"]
                sources = [f"{s['id']}: {s['text']}" for s in result["sources"]]
                full_reply = f"{reply.strip()}\n\n---\n**Sources:**\n" + "\n".join(f"• {s}" for s in sources[:3])
                st.markdown(full_reply)
                st.session_state.session_chat[-1]["bot"] = full_reply

    if len(st.session_state.session_chat) % 5 == 0:
        chat_history = [
            {"role": "user", "content": entry["user"]} if "user" in entry else {"role": "assistant", "content": entry["bot"]}
            for entry in st.session_state.session_chat
        ]
        await mem.summarize_procedural(chat_history, user_id=user_id)

    st.rerun()

if user_input:
    asyncio.run(handle_input(user_input))
