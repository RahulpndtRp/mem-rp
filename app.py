

import streamlit as st
import uuid
import os, json

os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

from my_mem.memory.main import Memory
from my_mem.configs.base import MemoryConfig



# -----------------------------
# 📦 Paths for profiles only
# -----------------------------
DATA_DIR = "user_data"
os.makedirs(DATA_DIR, exist_ok=True)
USERS_FILE = os.path.join(DATA_DIR, "users.json")

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
    if os.path.exists(USERS_FILE):
        st.session_state.users = load_users()
    else:
        default_user = f"u-{uuid.uuid4().hex[:6]}"
        st.session_state.users = [default_user]
        save_users(st.session_state.users)


if "selected_user" not in st.session_state:
    st.session_state.selected_user = st.session_state.users[0]

if "mem" not in st.session_state:
    st.session_state.mem = Memory(MemoryConfig())
mem = st.session_state.mem

if "session_chat" not in st.session_state:
    st.session_state.session_chat = []

# -----------------------------
# 👤 Sidebar: User Management
# -----------------------------
with st.sidebar:
    st.markdown("### 🧑‍💻 Select or Add User")
    user = st.selectbox("Active User ID", st.session_state.users, index=st.session_state.users.index(st.session_state.selected_user))

    if st.button("➕ Add New User"):
        new_id = f"u-{uuid.uuid4().hex[:6]}"
        st.session_state.users.append(new_id)
        st.session_state.selected_user = new_id
        save_users(st.session_state.users)
        st.session_state.session_chat = []  # reset session chat
        st.rerun()

    if user != st.session_state.selected_user:
        st.session_state.selected_user = user
        st.session_state.session_chat = []  # reset session chat
        st.rerun()

    st.markdown("---")
    st.code(f"User ID: {st.session_state.selected_user}")
    st.success("Short-term & Long-term memory enabled")

# -----------------------------
# 💬 Chat Display
# -----------------------------
st.markdown("## 🧠 Memory Chat Interface")

for chat in st.session_state.session_chat:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat.get("bot", "_Thinking..._"))

# -----------------------------
# ✍️ Input Box
# -----------------------------
user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.session_chat.append({"user": user_input})
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔄 Thinking..."):
            user_id = st.session_state.selected_user

            # Backend handles memory
            mem.add(user_input, user_id=user_id, infer=True)
            results = mem.search(user_input, user_id=user_id)["results"]
            context = "\n".join([r["memory"] for r in results])
            sources = [f"{r['id']}: {r['memory']}" for r in results]

            prompt = f"Context:\n{context}\n\nQuestion: {user_input}"
            reply = mem.llm.generate_response(messages=[{"role": "user", "content": prompt}])

            st.session_state.session_chat[-1]["bot"] = f"{reply.strip()}\n\n---\n**Sources:**\n" + "\n".join(f"• {s}" for s in sources[:3])
            st.rerun()
