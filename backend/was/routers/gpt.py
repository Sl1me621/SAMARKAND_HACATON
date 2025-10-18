# backend/routers/gpt.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.core.inference import gpt_inference

router = APIRouter(prefix="/api/gpt", tags=["GPT"])

class ChatIn(BaseModel):
    message: str = Field(..., min_length=1)

class ChatOut(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    try:
        return ChatOut(answer=gpt_inference(body.message))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gpt error: {e}")
