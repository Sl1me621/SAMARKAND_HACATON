# backend/routers/llama.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.core.inference import llama_inference

router = APIRouter(prefix="/api/llama", tags=["LLaMA"])

class ChatIn(BaseModel):
    message: str = Field(..., min_length=1)

class ChatOut(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    try:
        return ChatOut(answer=llama_inference(body.message))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"llama error: {e}")
