# frontend/app.py
import streamlit as st
import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Samarkand Agent",
    page_icon="💬",
    layout="centered"
)

st.markdown("""
<style>
    body { font-family: Arial; background-color: white; }
    .user-msg { background-color: #F0F8FF; color: grey; padding: 8px; margin: 5px; text-align: right; border-radius: 5px; }
    .bot-msg { background-color: #E6F3FF; color: grey; padding: 8px; margin: 5px; text-align: left; border-radius: 5px; }
    .title { color: black; text-align: center; font-size: 28px; font-weight: bold; }
    .stMarkdown { color: #333333; }
    .stTextInput>div>div>input { color: #333333; background-color: white; border: 1px solid #cccccc; }
    .stButton>button { color: white; background-color: #4169E1; border: none; }
    .stButton>button:hover { background-color: #2E5AC1; }
    .question-label { color: black; font-weight: bold; font-size: 16px; }
    .stTextInput>div>div>input::placeholder { color: #000000; opacity: 1; }
    .footer { color: #000000; text-align: center; padding: 20px; font-size: 14px; position: fixed; bottom: 0; left: 0; right: 0; background-color: white; border-top: 1px solid #e0e0e0; z-index: 1000; }
    .stApp { background-color: white; min-height: 100vh; position: relative; padding-bottom: 60px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Samarkand Agent</div>', unsafe_allow_html=True)
st.write("Бот поддержки")

# История чата
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Привет! Я твой помощник. Задай мне вопрос!"}
    ]

# Рендер сообщений
st.markdown("---")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-msg"><b>Вы:</b> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg"><b>Бот:</b> {message["content"]}</div>', unsafe_allow_html=True)

# Поле ввода
st.markdown("---")
st.write("**Напиши сообщение:**")
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="question-label">Твой вопрос:</div>', unsafe_allow_html=True)
    user_text = st.text_input(
        "Вопрос",
        placeholder="Введите свой вопрос",
        label_visibility="collapsed",
        key="user_text"
    )

with col2:
    send_btn = st.button("Отправить")

def ask_backend(text: str) -> str:
    """Бьёмся в FastAPI /api/message и возвращаем answer либо ошибку."""
    try:
        url = f"{API_URL}/api/message"
        resp = requests.post(url, json={"text": text}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "ok":
            return data.get("answer", "Пустой ответ")
        else:
            return f"Ошибка сервера: {data.get('meta', {}).get('error', 'unknown')}"
    except requests.exceptions.RequestException as e:
        return f"Не удалось подключиться к бэку: {e}"

# Обработка сообщения
if send_btn and user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.spinner("Думаю..."):
        time.sleep(0.1)
        bot_answer = ask_backend(user_text)
    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
    st.rerun()

st.markdown('<div class="footer">Samarkand agent 2025, МИРЭА</div>', unsafe_allow_html=True)
