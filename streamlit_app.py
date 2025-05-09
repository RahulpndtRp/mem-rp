import streamlit as st
import uuid, os, json

from my_mem.client import MemoryClient
from my_mem.configs.base import MemoryConfig

# Set OpenAI key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

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
# 🧠 Memory Init
# -----------------------------
st.set_page_config(page_title="Memory Chat", layout="wide")

if "users" not in st.session_state:
    st.session_state.users = load_users() or [f"Agent-{uuid.uuid4().hex[:4]}"]
    save_users(st.session_state.users)

if "selected_user" not in st.session_state:
    st.session_state.selected_user = st.session_state.users[0]

if "mem" not in st.session_state:
    st.session_state.mem = MemoryClient(MemoryConfig())
mem = st.session_state.mem

if "session_chat" not in st.session_state:
    st.session_state.session_chat = []

# -----------------------------
# 🧠 Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 👤 Select User")
    user = st.selectbox("User ID", st.session_state.users, index=st.session_state.users.index(st.session_state.selected_user))

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

    st.markdown("---")
    stream_enabled = st.checkbox("🔁 Enable Streaming", value=True)
    st.code(f"User ID: {st.session_state.selected_user}")
    st.success("Short-term + Long-term memory is active")

    if st.button("🗑️ Clear All Users"):
        st.session_state.users.clear()
        st.session_state.selected_user = None
        st.session_state.session_chat = []
        save_users([])
        st.rerun()

# -----------------------------
# 💬 Display Chat
# -----------------------------
st.markdown("## 💬 Chat Memory Agent")

for msg in st.session_state.session_chat:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg.get("bot", "_Thinking..._"))

# -----------------------------
# ⌨️ Chat Input
# -----------------------------
user_input = st.chat_input("Type a message...")
if user_input:
    user_id = st.session_state.selected_user
    st.session_state.session_chat.append({"user": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if stream_enabled:
            with st.spinner("🧠 Thinking (streaming)..."):
                response_container = st.empty()
                full_response = ""

                for token in mem.stream_rag(user_input, user_id=user_id):
                    full_response += token
                    response_container.markdown(full_response + "▌")  # typing indicator

                response_container.markdown(full_response.strip())
                st.session_state.session_chat[-1]["bot"] = full_response.strip()
        else:
            with st.spinner("🧠 Thinking..."):
                result = mem.query_rag(user_input, user_id=user_id)
                reply = result["answer"]
                sources = [f"{s['id']}: {s['text']}" for s in result["sources"]]

                full_reply = f"{reply.strip()}\n\n---\n**Sources:**\n" + "\n".join(f"• {s}" for s in sources[:3])
                st.markdown(full_reply)
                st.session_state.session_chat[-1]["bot"] = full_reply

    st.rerun()
