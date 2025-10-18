# backend/core/agent.py
import datetime
import re
from typing import Any, Dict, List
import requests

# Карта доступных роутеров (эндпоинтов FastAPI)
MODEL_ENDPOINTS: Dict[str, str] = {
    "mock":   "http://127.0.0.1:8000/api/mock/chat",
    "deepsy": "http://127.0.0.1:8000/api/deepsy/chat",
    "gpt":    "http://127.0.0.1:8000/api/gpt/chat",
    "llama":  "http://127.0.0.1:8000/api/llama/chat",
}

class SamarkandAgent:
    def __init__(self) -> None:
        self.memory: List[Dict[str, Any]] = []

    # ---------- utils ----------
    def time_now(self) -> str:
        return datetime.datetime.now().astimezone().isoformat()

    def log_event(self, event: Dict[str, Any]) -> None:
        self.memory.append({
            "time": self.time_now(),
            "type": "log",
            "event": event
        })

    # ---------- классификация (заглушка) ----------
    def classifier(self, text: str) -> Dict[str, Any]:
        """
        Эвристическая заглушка выбора модели.
        Правила:
        - 'строг', 'без домысл', 'точн'  -> deepsy (строгий тон)
        - 'mock' или 'тест'              -> mock (заглушка)
        - 'gpt'                           -> gpt
        - длина текста >= 200            -> llama
        - иначе                           -> deepsy
        """
        t = (text or "").lower()

        if re.search(r"строг|без домысл|точн", t):
            cls = "deepsy"
        elif re.search(r"\b(mock|тест)\b", t):
            cls = "mock"
        elif "gpt" in t:
            cls = "gpt"
        elif len(t) >= 200:
            cls = "llama"
        else:
            cls = "deepsy"

        return {"class": cls, "reason": "heuristic_stub"}

    def classify_task(self, text: str) -> str:
        return self.classifier(text)["class"]

    # ---------- роутинг ----------
    def route_task(self, task_type: str) -> str:
        """
        Возвращает URL эндпоинта выбранной модели.
        По умолчанию — deepsy.
        """
        return MODEL_ENDPOINTS.get(task_type, MODEL_ENDPOINTS["deepsy"])

    # ---------- единая точка входа ----------
    def ask_samarkand(self, question: str) -> str:
        """
        1) Классифицируем запрос (заглушка)
        2) Выбираем модель -> эндпоинт
        3) POST на соответствующий роутер
        4) Возвращаем answer
        """
        text = (question or "").strip()
        if not text:
            return "Пустое сообщение"

        # 1) классификация
        model_key = self.classify_task(text)
        endpoint = self.route_task(model_key)

        # 2) лог решения
        self.log_event({
            "kind": "route_decision",
            "model": model_key,
            "endpoint": endpoint,
            "question": text
        })

        # 3) запрос к выбранному роутеру
        payload: Dict[str, Any] = {"message": text}
        if model_key == "deepsy":
            payload["temperature"] = 0.2  # демонстрация параметра

        try:
            resp = requests.post(endpoint, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "Нет ответа")
            else:
                answer = f"Ошибка {resp.status_code} от {model_key}"
        except Exception as e:
            answer = f"Ошибка обращения к {model_key}: {e}"

        # 4) лог ответа
        self.log_event({
            "kind": "route_result",
            "model": model_key,
            "endpoint": endpoint,
            "answer": answer
        })

        return answer

    # ---------- сервисная инфа ----------
    def get_info(self) -> Dict[str, Any]:
        return {
            "agent_name": "SamarkandAgent",
            "version": "0.2",
            "history_length": len(self.memory),
        }
