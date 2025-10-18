# # внутри активного venv
pip install fastapi uvicorn[standard] pydantic python-dotenv requests streamlit



uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
http://localhost:8000/api/health


python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
streamlit run frontend/app.py

# Samarkand Agent — Backend + RAG‑ready

> Быстрый старт и понятные инструкции для запуска локально и на сервере. Один цельный README без разлёта по файлам.

---

## 1) Что это и как работает (в 3 фразы)

* Принимаем сообщение пользователя → классифицируем по сложности (S1/S2/S3).
* S1: отвечаем из локальной базы знаний (FAQ/каталог ответов).
* Если S1 не смог или низкая уверенность → S2 (внешние LLM: DeepSeek + GigaChat, оркестрация) → если всё ещё плохо/эскалация → S3 (человек‑оператор).

### Быстрая схема потока

`frontend → /api/chat → classifier(S1?) → retriever(DB/embeddings) → answer`
`else → external LLMs (DeepSeek+GigaChat, строгий режим) → answer`
`else → human_handoff → ticket_id`

---

## 2) Структура репозитория

```
.
├─ backend/                # FastAPI приложение (заменённая версия V2)
│  ├─ main.py              # Точка входа FastAPI (uvicorn)
│  ├─ clients/
│  │  └─ deepseek.py       # Клиент для DeepSeek (HTTP)
│  ├─ routers/ (опц.)      # Если используем роутеры: /api/gpt, /api/mock и т.п.
│  ├─ core/                # Логика: классификация, ранжирование, трейсинг, guardrails
│  └─ ...
├─ knowledge/              # Данные базы знаний (FAQ, CSV, JSON, Markdown)
├─ models/                 # Локальные модели/веса (если нужны)
├─ rosatom_ml/             # Наброски/эксперименты ML (если нужны)
├─ main.py                 # (если был исторический файл запуска) можно удалить/или прокинуть на backend
├─ requirements.txt        # Зависимости бэка (обновлены под V2)
├─ .env                    # Локальные секреты (НЕ пушим)
├─ .env.example            # Пример env (можно пушить)
└─ README.md               # Этот файл
```

> Если в репо есть `main.py` в корне — либо удаляем, либо оставляем, но явно пишем в нём «deprecated» и перенаправляем к `backend/main.py`.

---

## 3) Быстрый старт (локально)

### 3.1 Установка

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 Настроить переменные окружения

Создай `.env` в корне (рядом с `requirements.txt`) и заполни:

```
# обязательные
DEEPSEEK_API_KEY=...            # ключ DeepSeek (иначе будет ошибка "DEEPSEEK_API_KEY is not set")
# опционально, если включаем агрегирование
GIGACHAT_API_KEY=...
GIGACHAT_BASE_URL=https://gigachat.devices.sberbank.ru/api/v1 # пример

# базовые настройки
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
ENV=local

# RAG (когда подключим)
EMBEDDINGS_MODEL=text-embeddings-mini  # пример
VECTOR_DB_PATH=.data/index             # локальный путь под векторный индекс
CONFIDENCE_THRESHOLD=0.65              # порог S1 → S2
STRICT_MODE=true                       # строгие ответы внешних LLM
```

> Рекомендуется закоммитить `.env.example` (без секретов), а `.env` добавить в `.gitignore`.

### 3.3 Запуск

```bash
uvicorn backend.main:app --reload --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8000}
```

Проверка:

* `GET http://localhost:8000/ping` → `{ "status": "ok" }`
* `POST http://localhost:8000/api/chat` → тело `{ "message": "Привет" }`

---

## 4) Эндпоинты

* `GET /ping` — healthcheck.
* `POST /api/chat` — основной входной эндпоинт.

  * Внутри: логирование → классификация → S1/S2/S3.
  * Ответ: `{ answer, source, level, score, trace_id }`.
* `POST /api/feedback` — опционально: собираем фидбек для обучения.

---

## 5) Классификация и маршрутизация (кратко, как проверяют на защите)

1. **S1 (Локалка из БД)**: берём вопрос → векторизация → поиск ближайших пассажа/FAQ → формируем краткий ответ.

   * Если `score < CONFIDENCE_THRESHOLD` → переходим в S2.
2. **S2 (Внешние LLM, 2 модели)**: параллельные запросы к DeepSeek и GigaChat в строгом стиле (system/prompt «кратко, без фантазий, если не знаешь — скажи, что не знаешь»).

   * Сверяем ответы (консенсус/ранжирование по эвристике).
   * Если всё плохо → S3.
3. **S3 (Человек)**: создаём тикет, возвращаем `ticket_id` и шаблонный ответ.

> Логи каждого шага пишем в stdout и (опц.) в файл `./logs/app.log`.

---

## 6) RAG‑интеграция (пока не установлено)

Поддержка уже учтена в кодовой структуре, но по умолчанию выключена.

**Как подключить:**

1. Добавить в `requirements.txt`: `faiss-cpu` или `chromadb`, `sentence-transformers` (или аналоги из выбранного стека).
2. Подготовить скрипт индексации: `python scripts/build_index.py knowledge/ .data/index`.
3. В рантайме при `S1` использовать retriever на основе индекса.
4. Для обновлений знаний: CLI `python scripts/refresh_index.py` (перестройка по расписанию/хуку).

**Мини‑чеклист:**

* [ ] `.env` переменные для эмбеддингов и пути к индексу.
* [ ] health‑проверка наличия индекса при старте.
* [ ] graceful fallback: если индекс не найден → лог + сразу S2.

---

## 7) Требования (requirements.txt)

Минимально (пример):

```
fastapi
uvicorn[standard]
httpx
pydantic
python-dotenv
loguru
# для RAG потом:
# chromadb
# faiss-cpu
# sentence-transformers
```

---

## 8) Тестовые запросы (curl)

curl -s http://localhost:8000/ping

# чат
curl -s -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Как оформить карту москвича?"}' | jq

## 9 Деплой (кратко)

### Systemd (bare‑metal)

```
[Unit]
Description=SamarkandAgent API
After=network.target

[Service]
WorkingDirectory=/opt/samarkand
EnvironmentFile=/opt/samarkand/.env
ExecStart=/opt/samarkand/.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

### Docker (опционально)

```
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 10) Траблшутинг

* **DeepSeek request failed: DEEPSEEK_API_KEY is not set**
  → Добавь ключ в `.env`. Проверь, что процесс видит переменные (перезапусти uvicorn/systemd).

* **'latin-1' codec can't encode characters ...**
  → Причина — вывод/лог с кириллицей в не‑UTF8 окружении. Решения:

  1. Запуск с `PYTHONIOENCODING=utf-8`
  2. В Windows PowerShell: `chcp 65001` перед запуском
  3. Убедись, что логгер пишет UTF‑8 (`encoding='utf-8'`).

* **CORS ошибки на фронте**
  → Включи `fastapi.middleware.cors` и добавь домен фронта в allow‑list.

---

## 11) Политика ответов (строгий режим)

System‑prompt для внешних моделей:

* «Отвечай кратко и уверенно, без фантазий.
* Если данных недостаточно — скажи: “Недостаточно данных, передаю запрос оператору”.
* Никаких ссылок на несуществующие источники».

---

## 12) Вклад / как развивать

* Пул‑реквесты в ветке `feat/*`.
* Линтеры/форматирование (black/isort) — по желанию.
* Описываем изменения в CHANGELOG или в теле PR.

---

## 13) Контрольный список перед демо

* [ ] `/ping` отвечает
* [ ] `POST /api/chat` отрабатывает S1 на простых FAQ из `knowledge/`
* [ ] При низком score уходит в S2 и получает ответ от внешних LLM
* [ ] S3 отдаёт `ticket_id` и логируется
* [ ] Логи понятны и читаемы
* [ ] `.env.example` присутствует, `.env` не в гите