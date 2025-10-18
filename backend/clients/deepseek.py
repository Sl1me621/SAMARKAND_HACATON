# backend/clients/deepseek.py
from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

from dotenv import load_dotenv
from pathlib import Path

ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ROOT_ENV)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or ""
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


class DeepseekError(RuntimeError):
    pass

def _headers() -> Dict[str, str]:
    if not DEEPSEEK_API_KEY:
        raise DeepseekError("DEEPSEEK_API_KEY is not set")
    return {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

def _post_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    """POST с простым ретраем на 429/5xx."""
    backoff = 0.7
    for attempt in range(4):
        try:
            resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise DeepseekError(f"Transient HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, DeepseekError) as e:
            if attempt == 3:
                raise DeepseekError(f"DeepSeek request failed: {e}")
            time.sleep(backoff)
            backoff *= 1.8
    raise DeepseekError("Unexpected retry loop exit")

def chat(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout: int = 30,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Делает вызов /v1/chat/completions и возвращает (text, full_json).
    messages — массив dict вида {"role": "system"|"user"|"assistant", "content": "..."}.
    """
    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model or DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)

    data = _post_json(url, payload, timeout=timeout)

    
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        raise DeepseekError(f"Unexpected response schema: {data}")
    return text, data

def ask(
    user_text: str,
    *,
    system: str = (
        "You are a customer support assistant. Answer briefly, clearly, and definitively. "
        "If information is insufficient, say 'Недостаточно данных' and request one missing detail. "
        "Do not invent facts."
    ),
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """Упрощённый хелпер: одна реплика пользователя → ответ модели."""
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    text, _ = chat(msgs, temperature=temperature, max_tokens=max_tokens)
    return text
