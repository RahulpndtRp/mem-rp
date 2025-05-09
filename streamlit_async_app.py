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

# -----------------------------
# 📁 User Directory
# -----------------------------
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

# -----------------------------
# ✅ Streamlit Session Init
# -----------------------------
st.set_page_config(page_title="Memory Chat", layout="wide")

if "users" not in st.session_state:
    st.session_state.users = load_users()
    if not st.session_state.users:
        default_user = f"Agent-{uuid.uuid4().hex[:4]}"
        st.session_state.users = [default_user]
        save_users(st.session_state.users)

if "selected_user" not in st.session_state or st.session_state.selected_user not in st.session_state.users:
    if st.session_state.users:
        st.session_state.selected_user = st.session_state.users[0]
    else:
        st.session_state.selected_user = f"Agent-{uuid.uuid4().hex[:4]}"
        st.session_state.users = [st.session_state.selected_user]
        save_users(st.session_state.users)

if "mem" not in st.session_state:
    st.session_state.mem = AsyncMemoryClient(config)

if "session_chat" not in st.session_state:
    st.session_state.session_chat = []

mem: AsyncMemoryClient = st.session_state.mem

# -----------------------------
# 👤 Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 👤 Select User")
    selected_index = st.session_state.users.index(st.session_state.selected_user)
    user = st.selectbox("User ID", st.session_state.users, index=selected_index)

    if st.button("➕ Add User"):
        new_user = f"Agent-{uuid.uuid4().hex[:4]}"
        st.session_state.users.append(new_user)
        st.session_state.selected_user = new_user
        save_users(st.session_state.users)
        st.session_state.session_chat = []
        st.rerun()

    if user != st.session_state.selected_user:
        st.session_state.selected_user = user
        st.session_state.session_chat = []
        st.rerun()

    stream_enabled = st.checkbox("🔁 Enable Streaming", value=True)
    st.code(f"User ID: {st.session_state.selected_user}")
    st.success("Short-term + Long-term memory is active")

    if st.button("🗑️ Clear All Users"):
        st.session_state.show_popup = True

    if st.session_state.get("show_popup", False):
        st.warning("⚠️ This action is irreversible.")
        col1, col2 = st.columns(2)
        if col1.button("✅ Confirm"):
            st.session_state.users = []
            st.session_state.selected_user = ""
            st.session_state.session_chat = []
            save_users([])
            st.session_state.show_popup = False
            st.rerun()
        if col2.button("❌ Cancel"):
            st.session_state.show_popup = False
        st.rerun()

# -----------------------------
# 💬 Display Chat
# -----------------------------
st.markdown("## 💬 Memory Chat")

for msg in st.session_state.session_chat:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg.get("bot", "_Thinking..._"))

# -----------------------------
# ⌨️ Async Input Handler
# -----------------------------
user_input = st.chat_input("Type your message here...")

async def handle_input(user_input: str):
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

    st.rerun()

if user_input:
    asyncio.run(handle_input(user_input))
