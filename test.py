import requests

BASE = "http://127.0.0.1:8000"
zapros = input("Введи запрос (например, ping): ").strip().lstrip("/")
url = f"{BASE}/{zapros}" if zapros else f"{BASE}/"

try:
    r = requests.get(url, timeout=10)
    print("Статус:", r.status_code)
    try:
        print("Ответ:", r.json())
    except ValueError:
        print("Ответ:", r.text)
except requests.RequestException as e:
    print("Ошибка запроса:", e)