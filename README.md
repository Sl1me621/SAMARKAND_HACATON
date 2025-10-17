# SAMARKAND_HACATON
# Samarkand API (FastAPI)

Мини-бэкенд для демо: единый класс SamarkandAgent + HTTP API.

## Требования
- Python 3.10+ (рекомендовано 3.11)
- Git
- Интернет для установки зависимостей

## Клонирование
```bash
git clone https://github.com/Sl1me621/SAMARKAND_HACATON.git
cd SAMARKAND_HACATON

windows

python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt


linux

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt


start
uvicorn backend.main:app --host 0.0.0.0 --port 8000



---

# Проверки перед презентацией (у вас на месте)

1. Активировать venv и поставить зависимости по `requirements.txt`.  
2. `uvicorn backend.main:app --reload` → открыть http://127.0.0.1:8000/docs.  
3. Для доступа из сети: `--host 0.0.0.0` → открыть `http://<ваш_IP>:8000/docs` с другого ПК.  
4. Если используете VPN — на время демо лучше отключить.  

---

Если хочешь, пришли коротко содержимое `backend/main.py` — проверю, что там правильно импортируется `SamarkandAgent` и объявлен `app`, чтобы точно не схлопотать `Attribute "app" not found`.
