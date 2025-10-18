from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = FastAPI(title="UltraSimpleBackend")

# ====== МОДЕЛИ ВВОД/ВЫВОД ======
class MessageIn(BaseModel):
    text: str
    # можно прислать желаемую "сложность", если на фронте есть переключатель:
    level: Optional[Literal[1, 2, 3]] = None  # 1=локалка, 2=внешние API, 3=человек

class MessageOut(BaseModel):
    answer: str
    route: Literal["local", "external", "human"]

# ====== ОДНА ТОЧКА ВХОДА ======
@app.post("/ask", response_model=MessageOut)
def ask(payload: MessageIn):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Пустое сообщение")

    # 1) элементарная «классификация» без ML (реально по-тупому)
    level = payload.level or classify(text)

    # 2) роутинг в одну из трёх простых функций
    if level == 1:
        answer = handle_local(text)
        route = "local"
    elif level == 2:
        answer = handle_external(text)
        route = "external"
    else:
        answer = handle_human(text)
        route = "human"

    return MessageOut(answer=answer, route=route)

# ====== «КЛАССИФИКАЦИЯ» БЕЗ МАГИИ ======
def classify(text: str) -> int:
    t = text.lower()
    # примитивные правила; поменяешь под свой домен
    if any(k in t for k in ["цена", "тариф", "гарантия", "как настроить", "инструкция"]):
        return 1  # есть ответ в локальной БД/FAQ
    if any(k in t for k in ["анализ", "сравни", "подбери", "объясни подробно"]):
        return 2  # просим внешние модели
    return 3  # остальное — человеку

# ====== ОБРАБОТЧИКИ ======
def handle_local(text: str) -> str:
    # тут типа «локальная БД» — сейчас имитация словарём
    faq = {
        "тариф": "Наш базовый тариф — 499₽/мес. Подробности в личном кабинете.",
        "гарантия": "Гарантия 12 месяцев при наличии чека."
    }
    for k, v in faq.items():
        if k in text.lower():
            return v
    return "Нашёлся общий ответ из локальной базы: проверьте раздел FAQ."

def handle_external(text: str) -> str:
    # никаких ключей — просто заглушка; где нужно — вот тут подключишь API
    # пример: use_gigachat(text) или use_deepsy(text)
    # сейчас — возвращаем аккуратный шаблон
    return f"(Внешняя модель) Короткий уверенный ответ по запросу: «{text}»."

def handle_human(text: str) -> str:
    # имитация постановки в очередь оператору
    return "Ваш вопрос передан специалисту. Ответим в ближайшее рабочее время."

# ====== ЛОКАЛЬНЫЙ ЗАПУСК ======
# uvicorn backend.main:app --reload
