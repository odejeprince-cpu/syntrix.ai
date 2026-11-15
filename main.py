from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

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
    # FAKE reply for now — real AI comes in Step 4
    return {"reply": f"You said: {payload.message}"}