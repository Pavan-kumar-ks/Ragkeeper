import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from ragkeeper.config import get_settings
from ragkeeper.rag_chain import RagKeeperChat
from ragkeeper.vectorstore import get_client

from dashboard import analytics, chat, evaluation, health

st.set_page_config(page_title="RAGKeeper", layout="wide")
st.title("RAGKeeper")


@st.cache_resource
def _load_chat() -> RagKeeperChat:
    return RagKeeperChat()


settings = get_settings()
chat_service = _load_chat()
client = get_client(settings.qdrant_url)

tab_chat, tab_health, tab_eval, tab_analytics = st.tabs(
    ["Chat", "Index Health", "Evaluation", "Analytics"]
)

with tab_chat:
    chat.render(chat_service, settings)

with tab_health:
    health.render(settings, client)
    st.divider()
    health.render_system_health(settings, client)

with tab_eval:
    evaluation.render()

with tab_analytics:
    analytics.render(settings)
