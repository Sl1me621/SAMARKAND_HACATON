# backend/core/inference.py
def mock_inference(text: str) -> str:
    return f"[MOCK] понял: {text}"

def gpt_inference(text: str) -> str:
    # TODO: здесь будет реальный вызов внешнего GPT-провайдера
    # например через их HTTP API
    return f"[GPT] сгенерировал: {text}"

def llama_inference(text: str) -> str:
    # TODO: здесь будет вызов локальной/серверной LLaMA
    return f"[LLAMA] ответил: {text}"

def deepsy_inference(text: str, temperature: float = 0.2) -> str:
    """
    Заглушка "Дипсика". Здесь позже будет реальный HTTP-вызов к провайдеру.
    Пока просто возвращает предсказуемую строку.
    """
    text = (text or "").strip()
    if not text:
        return "[DEEPSY MOCK] пустой запрос"
    return f"[DEEPSY MOCK] строгий ответ без домыслов: {text}"