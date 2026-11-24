import re
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

app = FastAPI()

# Serves everything in the static/ folder (UI images live in static/img/)
app.mount("/static", StaticFiles(directory="static"), name="static")

CODE_MODEL = "qwen2.5-coder:1.5b"
AUTO_MODEL = "qwen2.5:3b"
GENERAL_MODEL = "qwen2.5:7b"

# --- Answer length limits (in tokens; about 3 tokens = 2 words) ---
# Lower number = faster answers. Raise a number if replies get cut off too often.
CODE_MAX_TOKENS = 700
AUTO_MAX_TOKENS = 350
GENERAL_MAX_TOKENS = 300

# How long Ollama keeps a model in RAM after its last use ("30m", "2h", or -1 = never unload).
# Ollama's default is 5 minutes. Keeping it loaded avoids a 10-30 second reload after a pause.
KEEP_ALIVE = "30m"

CUT_OFF_NOTE = "\n\n(Answer cut short to save time. Reply \"continue\" to get the rest.)"

CODE_PROMPT = (
    "You are Syntrix, an expert programming assistant. "
    "Answer clearly. Return code in fenced blocks with the language tag. "
    "When debugging, explain the root cause before the fix. "
    "Be concise: give a short explanation and only the code that is needed. "
    "Write more only if the user asks for detail."
)

AUTO_PROMPT = (
    "You are Syntrix Auto, an expert automotive mechanical engineer. "
    "Use the provided reference context when relevant. "
    "If the context does not contain the answer, say so and give your best "
    "general engineering reasoning. Be precise and practical. "
    "Be concise: answer in about 120 words or fewer, using short sentences. "
    "Write more only if the user asks for detail."
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
    "'the sources do not say' instead of guessing. "
    "Be concise: answer in about 100 words or fewer, with no long lists and "
    "no introduction or summary. Write more only if the user asks for detail."
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

# Words that show a question refers back to an earlier topic ("when was HE born?")
PRONOUNS = {"he", "she", "it", "its", "they", "them", "their", "his", "her",
            "him", "that", "this", "these", "those", "there"}
FOLLOWUP_STARTS = ("and ", "what about", "how about")

# --- RAG Setup (loaded once at startup) ---
print("Loading RAG system...")
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_chroma = chromadb.PersistentClient(path="chroma_db")
_amck = _chroma.get_collection("amck")
print(f"RAG ready: {_amck.count()} automotive Q&A pairs indexed.")
print("Syntrix follow-up search build v10 loaded")

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

_last_topic = ""  # the subject of the last standalone question, reused for follow-ups

def build_search_query(message: str) -> str:
    """Turns a message into a keyword query. For follow-ups like "when was he
    born?", it adds the topic of the last standalone question so the search
    has a subject."""
    global _last_topic
    own = to_keywords(message)
    lowered = message.lower().strip()
    words = {w.strip("?.,!") for w in lowered.split()}
    is_followup = bool(words & PRONOUNS) or lowered.startswith(FOLLOWUP_STARTS)

    if is_followup and _last_topic:
        merged = []
        for w in (_last_topic + " " + own).split():
            if w.lower() not in [m.lower() for m in merged]:
                merged.append(w)
        query = " ".join(merged[:8])
        print(f"Follow-up detected: '{message}' -> search '{query}'")
        return query

    # Standalone question: remember its subject (capitalized words if any)
    if not is_followup:
        caps = [w for w in own.split() if w[:1].isupper()]
        _last_topic = " ".join(caps[:4]) if caps else " ".join(own.split()[:4])
    return own

def web_search(query: str, max_results: int = 4):
    """query is already a keyword string. Tries the full query first, then
    shorter versions of it (the free search source sometimes returns nothing
    for oddly worded queries). Returns a list of dicts with title, body, href,
    or an empty list if every attempt fails."""
    words = query.split()
    candidates = []
    for n in (len(words), 3, 2):
        q = " ".join(words[:n])
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
        time.sleep(2)
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

def save_to_history(user_text: str, reply: str):
    """Keeps only the last HISTORY_LIMIT messages (no warnings or sources)."""
    global _history
    _history.append({"role": "user", "content": user_text})
    _history.append({"role": "assistant", "content": reply})
    if len(_history) > HISTORY_LIMIT:
        _history = _history[-HISTORY_LIMIT:]

def prepare(payload: ChatIn):
    """Does everything that happens BEFORE the model starts writing:
    picks the model, gathers RAG / web context, builds the messages list.
    Returns (model, messages, options, warning_text, sources_text)."""
    sources_text = ""
    warning_text = ""

    if payload.mode == "auto":
        context = retrieve_context(payload.message)
        system = AUTO_PROMPT + f"\n\n--- Reference material ---\n{context}\n--- End reference ---"
        model = AUTO_MODEL
        options = {"num_predict": AUTO_MAX_TOKENS}
    elif payload.mode == "general":
        results = web_search(build_search_query(payload.message))
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
        options = {"temperature": 0.2, "num_ctx": 4096, "num_predict": GENERAL_MAX_TOKENS}
    else:
        system = CODE_PROMPT
        model = CODE_MODEL
        options = {"num_predict": CODE_MAX_TOKENS}

    messages = [{"role": "system", "content": system}]
    messages.extend(_history[-HISTORY_LIMIT:])
    messages.append({"role": "user", "content": payload.message})
    return model, messages, options, warning_text, sources_text

@app.get("/")
def home():
    return {"message": "Hello from Syntrix"}

@app.get("/chat")
def chat_page():
    return FileResponse("static/chat.html")

@app.post("/chat/clear")
def clear_chat():
    global _history, _last_topic
    _history = []
    _last_topic = ""
    return {"status": "cleared"}

# Non-streaming version (waits for the full answer). Kept for testing in /docs.
@app.post("/chat")
def chat_reply(payload: ChatIn):
    try:
        model, messages, options, warning_text, sources_text = prepare(payload)
        resp = ollama.chat(
            model=model, messages=messages, options=options, keep_alive=KEEP_ALIVE
        )
        reply = resp["message"]["content"]
        save_to_history(payload.message, reply)
        cut = CUT_OFF_NOTE if getattr(resp, "done_reason", None) == "length" else ""
        return {"reply": warning_text + reply + cut + sources_text}
    except Exception as e:
        return {"reply": f"⚠️ AI error: {e}"}

# Streaming version: sends the answer piece by piece as the model writes it.
@app.post("/chat/stream")
def chat_stream(payload: ChatIn):
    def generate():
        try:
            model, messages, options, warning_text, sources_text = prepare(payload)

            if warning_text:
                yield warning_text

            reply = ""
            cut_off = False
            for chunk in ollama.chat(
                model=model, messages=messages, options=options,
                stream=True, keep_alive=KEEP_ALIVE,
            ):
                piece = chunk["message"]["content"]
                if piece:
                    reply += piece
                    yield piece
                if getattr(chunk, "done_reason", None) == "length":
                    cut_off = True

            save_to_history(payload.message, reply)

            if cut_off:
                yield CUT_OFF_NOTE
            if sources_text:
                yield sources_text
        except Exception as e:
            yield f"\n\n⚠️ AI error: {e}"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
