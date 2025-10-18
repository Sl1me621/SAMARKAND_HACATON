# backend/routers/mock.py
from fastapi import APIRouter
from pydantic import BaseModel
from backend.core.inference import mock_inference

router = APIRouter(prefix="/api/mock", tags=["Mock"])

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    text = body.message.strip()
    if not text:
        return ChatOut(answer="Пустое сообщение")
    reply = mock_inference(text)
    return ChatOut(answer=reply)


