import streamlit as st
import uuid
import os, json

os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

from my_mem.client import MemoryClient
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
        default_user = f"Agent-{uuid.uuid4().hex[:4]}"
        st.session_state.users = [default_user]
        save_users(st.session_state.users)


if "selected_user" not in st.session_state:
    if st.session_state.users:
        st.session_state.selected_user = st.session_state.users[0]
    else:
        st.session_state.selected_user = None  # Or leave it unset


if "mem" not in st.session_state:
    st.session_state.mem = MemoryClient(MemoryConfig())
mem = st.session_state.mem

if "session_chat" not in st.session_state:
    st.session_state.session_chat = []

# -----------------------------
# 👤 Sidebar: User Management
# -----------------------------
with st.sidebar:
    st.markdown("### 🧑‍💼 Select or Add User")
    if st.session_state.users:
        selected_index = st.session_state.users.index(st.session_state.selected_user)
        user = st.selectbox("Active User ID", st.session_state.users, index=selected_index)
    else:
        st.warning("No users available. Please add a new user.")
        user = None

    if st.button("➕ Add New User"):
        new_id = f"Agent-{uuid.uuid4().hex[:4]}"
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

    # Button to trigger confirmation popup
    if st.button("🗑️ Clear All Users"):
        st.session_state.show_popup = True

    # Show the confirmation popup
    if st.session_state.get("show_popup", False):
        st.warning("⚠️ Are you sure you want to delete all users? This cannot be undone.")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Yes, Clear"):
                st.session_state.users = []
                st.session_state.selected_user = ""
                st.session_state.session_chat = []
                save_users([])  # Clear users.json
                st.success("✅ All users cleared.")
                st.session_state.show_popup = False
                st.rerun()

        with col2:
            if st.button("❌ Cancel"):
                st.session_state.show_popup = False
        st.rerun()



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
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔄 Thinking..."):
            user_id = st.session_state.selected_user
            mem.add_message(user_input, user_id=user_id, infer=True)
            rag_result = mem.query_rag(user_input, user_id=user_id)

            reply = rag_result["answer"]
            sources = [f"{s['id']}: {s['text']}" for s in rag_result["sources"]]

            st.session_state.session_chat[-1]["bot"] = f"{reply.strip()}\n\n---\n**Sources:**\n" + "\n".join(f"• {s}" for s in sources[:3])
            st.rerun()
