import re
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

app = FastAPI()

CODE_MODEL = "qwen2.5-coder:1.5b"
AUTO_MODEL = "qwen2.5:3b"
GENERAL_MODEL = "qwen2.5:7b"

CODE_PROMPT = (
    "You are Syntrix, an expert programming assistant. "
    "Answer clearly. Return code in fenced blocks with the language tag. "
    "When debugging, explain the root cause before the fix."
)

AUTO_PROMPT = (
    "You are Syntrix Auto, an expert automotive mechanical engineer. "
    "Use the provided reference context when relevant. "
    "If the context does not contain the answer, say so and give your best "
    "general engineering reasoning. Be precise and practical."
)

GENERAL_PROMPT = (
    "You are Syntrix General, a knowledgeable assistant covering science, "
    "math, engineering, history, technology, and everyday questions. "
    "Explain clearly, show your steps for math and reasoning problems, "
    "and say so honestly when you are unsure instead of guessing. "
    "When web search results are provided, base your answer ONLY on them "
    "for facts about people, events, numbers, names, and dates. Copy names, "
    "numbers, and dates exactly as written in the results. Do not combine "
    "or rearrange details across results unless they clearly agree. "
    "If the results do not contain the detail asked for, say "
    "'the sources do not say' instead of guessing. Keep answers concise."
)

NO_RESULTS_NOTE = (
    "\n\nIMPORTANT: Web search returned no results for this question. "
    "Do NOT state specific facts about people, events, numbers, names, or "
    "dates from memory, because they are likely to be wrong. Answer only "
    "general concepts you are confident about, and for specific facts say "
    "you could not verify them right now and suggest trying again."
)

NO_RESULTS_WARNING = (
    "⚠️ Web search returned no results, so this answer is from the model's "
    "memory only and may be wrong. Try rephrasing with fewer words.\n\n"
)

# Filler words removed when turning a question into search keywords
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "do", "does", "did", "have", "has", "had", "having",
    "how", "many", "much", "what", "whats", "who", "whom", "whose",
    "when", "where", "which", "why",
    "of", "to", "in", "on", "for", "and", "or", "at", "by", "with",
    "me", "my", "i", "you", "your", "it", "its", "they", "their", "them",
    "he", "she", "his", "her", "this", "that", "these", "those",
    "tell", "about", "please", "can", "could", "would", "should",
    "explain", "give", "know", "there",
}

# --- RAG Setup (loaded once at startup) ---
print("Loading RAG system...")
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_chroma = chromadb.PersistentClient(path="chroma_db")
_amck = _chroma.get_collection("amck")
print(f"RAG ready: {_amck.count()} automotive Q&A pairs indexed.")
print("Syntrix web search build v4 loaded")

def retrieve_context(query: str, k: int = 3) -> str:
    emb = _embedder.encode([query]).tolist()
    res = _amck.query(query_embeddings=emb, n_results=k)
    docs = res.get("documents", [[]])[0]
    return "\n\n".join(docs)

# --- Web search (free, no API key) ---
def to_keywords(message: str, max_words: int = 8) -> str:
    """Turn a natural-language question into a short keyword query."""
    words = re.findall(r"[A-Za-z0-9\.\-\+#]+", message)
    keywords = [w for w in words if w.lower() not in STOPWORDS]
    if not keywords:
        keywords = words
    return " ".join(keywords[:max_words])

def web_search(message: str, max_results: int = 4):
    """Returns a list of dicts with title, body, href. Empty list on failure."""
    full = to_keywords(message)
    short = " ".join(full.split()[:4])
    candidates = []
    for q in (full, short):
        if q and q not in candidates:
            candidates.append(q)

    for q in candidates:
        try:
            results = list(DDGS(timeout=15).text(q, max_results=max_results) or [])
            print(f"Web search: '{q}' -> {len(results)} results")
            if results:
                return results
        except Exception as e:
            print(f"Web search failed for '{q}': {e}")
        time.sleep(1)
    return []

def format_search_context(results) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        body = (r.get("body", "") or "")[:600]
        lines.append(f"[{i}] {title}\n{body}")
    return "\n\n".join(lines)

def format_sources(results) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        href = r.get("href", "")
        if href:
            lines.append(f"[{i}] {href}")
    if not lines:
        return ""
    return "\n\nSources:\n" + "\n".join(lines)

# --- Conversation memory ---
HISTORY_LIMIT = 10
_history = []

class ChatIn(BaseModel):
    message: str
    mode: str = "code"

@app.get("/")
def home():
    return {"message": "Hello from Syntrix"}

@app.get("/chat")
def chat_page():
    return FileResponse("static/chat.html")

@app.post("/chat/clear")
def clear_chat():
    global _history
    _history = []
    return {"status": "cleared"}

@app.post("/chat")
def chat_reply(payload: ChatIn):
    global _history
    try:
        sources_text = ""
        warning_text = ""
        options = {}

        if payload.mode == "auto":
            context = retrieve_context(payload.message)
            system = AUTO_PROMPT + f"\n\n--- Reference material ---\n{context}\n--- End reference ---"
            model = AUTO_MODEL
        elif payload.mode == "general":
            results = web_search(payload.message)
            system = GENERAL_PROMPT
            if results:
                system += (
                    "\n\n--- Web search results ---\n"
                    f"{format_search_context(results)}\n"
                    "--- End web search results ---"
                )
                sources_text = format_sources(results)
            else:
                system += NO_RESULTS_NOTE
                warning_text = NO_RESULTS_WARNING
            model = GENERAL_MODEL
            options = {"temperature": 0.2, "num_ctx": 4096}
        else:
            system = CODE_PROMPT
            model = CODE_MODEL

        messages = [{"role": "system", "content": system}]
        messages.extend(_history[-HISTORY_LIMIT:])
        messages.append({"role": "user", "content": payload.message})

        resp = ollama.chat(model=model, messages=messages, options=options)
        reply = resp["message"]["content"]

        # Save only the model's answer in history (not warnings or sources)
        _history.append({"role": "user", "content": payload.message})
        _history.append({"role": "assistant", "content": reply})
        if len(_history) > HISTORY_LIMIT:
            _history = _history[-HISTORY_LIMIT:]

        return {"reply": warning_text + reply + sources_text}
    except Exception as e:
        return {"reply": f"⚠️ AI error: {e}"}