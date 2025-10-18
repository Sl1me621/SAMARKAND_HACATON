from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from backend.clients.deepseek import ask as deepseek_ask, DeepseekError

APP_NAME = os.getenv("APP_NAME", "V2-App")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageIn(BaseModel):
    text: str

class MessageOut(BaseModel):
    status: str      # "ok" | "error"
    answer: str
    meta: dict | None = None

# ультра-простая классификация: 1 — локалка, 2 — внешка, 3 — человек
def classify(text: str) -> int:
    # t = text.lower()
    # if any(k in t for k in ["цена", "где", "как", "инструкция"]):
    #     return 1
    # if any(k in t for k in ["переведи", "сгенерируй", "объясни"]):
    #     return 2
    return 2

def handle_local(text: str) -> str:
    return "Ответ из локальной базы (заглушка)."

def handle_external(text: str) -> str:
    """Внешняя модель: DeepSeek."""
    try:
        # promtт для техподдержки
        return deepseek_ask(
            text,
            system=(
                "Ты ассистент техподдержки. Отвечай кратко и уверенно. "
                "Если данных мало — ответь 'Недостаточно данных' и назови ровно один ключевой уточняющий вопрос. "
                "Не выдумывай факты."
            ),
            temperature=0.1,
            max_tokens=400,
        )
    except DeepseekError as e:
        return f"Ошибка внешней модели: {e}"

def handle_human(text: str) -> str:
    return "Передано человеку-оператору (заглушка)."

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/message", response_model=MessageOut)
def message(payload: MessageIn):
    try:
        route = classify(payload.text)
        if route == 1:
            ans = handle_local(payload.text)
        elif route == 2:
            ans = handle_external(payload.text)
        else:
            ans = handle_human(payload.text)
        return MessageOut(status="ok", answer=ans, meta={"route": route})
    except Exception as e:
        return MessageOut(status="error", answer="internal error", meta={"error": str(e)})
