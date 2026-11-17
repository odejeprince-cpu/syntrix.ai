from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()

CODE_MODEL = "qwen2.5-coder:1.5b"
AUTO_MODEL = "qwen2.5:3b"

CODE_PROMPT = (
    "You are Syntrix, an expert programming assistant. "
    "Answer clearly. Return code in fenced blocks with the language tag. "
    "When debugging, explain the root cause before the fix."
)

AUTO_PROMPT = (
    "You are Syntrix Auto, an expert automotive mechanical engineer. "
    "You cover engines, transmissions, chassis, electrical systems, OBD-II "
    "diagnostics, maintenance, and repair. Use the provided reference context "
    "when relevant. If the context does not contain the answer, say so and "
    "give your best general engineering reasoning. Be precise and practical."
)

# --- RAG Setup (loaded once at startup) ---
print("Loading RAG system...")
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_chroma = chromadb.PersistentClient(path="chroma_db")
_amck = _chroma.get_collection("amck")
print(f"RAG ready: {_amck.count()} automotive Q&A pairs indexed.")

def retrieve_context(query: str, k: int = 3) -> str:
    emb = _embedder.encode([query]).tolist()
    res = _amck.query(query_embeddings=emb, n_results=k)
    docs = res.get("documents", [[]])[0]
    return "\n\n".join(docs)

class ChatIn(BaseModel):
    message: str
    mode: str = "code"

@app.get("/")
def home():
    return {"message": "Hello from Syntrix"}

@app.get("/chat")
def chat_page():
    return FileResponse("static/chat.html")

@app.post("/chat")
def chat_reply(payload: ChatIn):
    try:
        if payload.mode == "auto":
            context = retrieve_context(payload.message)
            system = AUTO_PROMPT + f"\n\n--- Reference material ---\n{context}\n--- End reference ---"
            model = AUTO_MODEL
        else:
            system = CODE_PROMPT
            model = CODE_MODEL

        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": payload.message},
            ],
        )
        return {"reply": resp["message"]["content"]}
    except Exception as e:
        return {"reply": f"⚠️ AI error: {e}"}