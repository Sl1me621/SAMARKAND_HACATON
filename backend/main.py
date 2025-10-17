from fastapi import FastAPI

app = FastAPI(title="Samarkand API")

@app.get("/")
def root():
    return {"message": "Hello, Samarkand!"}

@app.get("/ping")
def ping():
    return {"ok": True}
