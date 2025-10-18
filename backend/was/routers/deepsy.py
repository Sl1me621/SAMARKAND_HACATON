# backend/routers/deepsy.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.core.inference import deepsy_inference

router = APIRouter(prefix="/api/deepsy", tags=["Deepsy"])

class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, description="Пользовательский вопрос")
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="Параметр генерации (пока не используется)")

class ChatOut(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    try:
        reply = deepsy_inference(body.message, temperature=body.temperature)
        return ChatOut(answer=reply)
    except Exception as e:
        # чтобы фронту прилетела нормальная ошибка
        raise HTTPException(status_code=500, detail=f"deepsy error: {e}")
