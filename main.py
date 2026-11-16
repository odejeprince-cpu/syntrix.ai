from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import ollama

app = FastAPI()

CODE_MODEL = "qwen2.5-coder:1.5b"

SYSTEM_PROMPT = (
    "You are Syntrix, an expert programming assistant. "
    "Answer clearly. Return code in fenced blocks with the language tag. "
    "When debugging, explain the root cause before the fix."
)

class ChatIn(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Hello from Syntrix"}

@app.get("/chat")
def chat_page():
    return FileResponse("static/chat.html")

@app.post("/chat")
def chat_reply(payload: ChatIn):
    try:
        resp = ollama.chat(
            model=CODE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload.message},
            ],
        )
        return {"reply": resp["message"]["content"]}
    except Exception as e:
        return {"reply": f"⚠️ AI error: {e}\n\nIs `ollama serve` running?"}