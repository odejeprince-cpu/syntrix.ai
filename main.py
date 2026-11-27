import io
import re
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import ollama
import chromadb
from pypdf import PdfReader
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
# These are the "Normal" settings. Short and Long (chosen in Settings) scale them.
# Lower number = faster answers. Raise a number if replies get cut off too often.
CODE_MAX_TOKENS = 700
AUTO_MAX_TOKENS = 350
GENERAL_MAX_TOKENS = 300

# Settings > Answer length: short = half the tokens, long = double
LENGTH_SCALE = {"short": 0.5, "normal": 1.0, "long": 2.0}
LENGTH_HINTS = {
    "short": "\n\nThe user wants very short answers. Keep your answer to a few sentences.",
    "normal": "",
    "long": (
        "\n\nThe user wants a more detailed answer this time. Ignore any earlier "
        "instruction to be brief. Explain fully, with steps or examples where helpful."
    ),
}

def scaled(base: int, length: str) -> int:
    return max(100, int(base * LENGTH_SCALE.get(length, 1.0)))

# How long Ollama keeps a model in RAM after its last use ("30m", "2h", or -1 = never unload).
KEEP_ALIVE = "30m"

# --- File attachment limits ---
MAX_UPLOAD_BYTES = 5 * 1024 * 1024   # biggest file we accept (5 MB)
MAX_FILE_CHARS = 6000                # how much of the file the model reads (about 1500 tokens)
FILE_CTX = 6144                      # model memory size used when a file is attached
CUT_MARK = "\n[...the rest of the file was cut off...]"
TEXT_EXTENSIONS = {
    "txt", "md", "py", "json", "csv", "js", "ts", "html", "css", "log",
    "yml", "yaml", "xml", "ini", "toml", "sql", "java", "c", "cpp", "h", "sh",
}

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

# Used when the user switched web search OFF in Settings
SEARCH_OFF_NOTE = (
    "\n\nWeb search is turned off, so answer from your own knowledge. For specific "
    "facts about people, events, numbers, names, or dates that you are not sure of, "
    "say honestly that you could not verify them."
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

def retrieve_context(query: str, k: int = 3) -> str:
    emb = _embedder.encode([query]).tolist()
    res = _amck.query(query_embeddings=emb, n_results=k)
    docs = res.get("documents", [[]])[0]
    return "\n\n".join(docs)

# --- Chat storage (SQLite: one file, built into Python, no install) ---
DB_PATH = "syntrix.db"
HISTORY_LIMIT = 10  # how many recent messages the model gets as memory

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with closing(db()) as conn, conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                mode TEXT NOT NULL,
                topic TEXT NOT NULL DEFAULT '',
                created TEXT NOT NULL,
                updated TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,   -- what the model sees as memory
                display TEXT NOT NULL,   -- what the page shows (adds warnings, sources)
                created TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                name TEXT NOT NULL,
                text TEXT NOT NULL,      -- the text we pulled out of the file
                created TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_files_chat ON files(chat_id, id);
        """)

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def make_title(text: str) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= 45 else one_line[:44].rstrip() + "…"

def get_or_create_chat(chat_id: Optional[str], first_message: str, mode: str) -> str:
    with closing(db()) as conn, conn:
        if chat_id:
            row = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if row:
                return chat_id
        new_id = uuid.uuid4().hex[:12]
        t = now()
        conn.execute(
            "INSERT INTO chats (id, title, mode, created, updated) VALUES (?, ?, ?, ?, ?)",
            (new_id, make_title(first_message), mode, t, t),
        )
        return new_id

def load_context(chat_id: str):
    with closing(db()) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE chat_id = ? AND content != '' ORDER BY id DESC LIMIT ?",
            (chat_id, HISTORY_LIMIT),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def get_topic(chat_id: str) -> str:
    with closing(db()) as conn:
        row = conn.execute("SELECT topic FROM chats WHERE id = ?", (chat_id,)).fetchone()
    return row["topic"] if row else ""

def set_topic(chat_id: str, topic: str):
    with closing(db()) as conn, conn:
        conn.execute("UPDATE chats SET topic = ? WHERE id = ?", (topic, chat_id))

def add_message(chat_id: str, role: str, content: str, display: str, mode: str):
    t = now()
    with closing(db()) as conn, conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, display, created) "
            "VALUES (?, ?, ?, ?, ?)",
            (chat_id, role, content, display, t),
        )
        conn.execute("UPDATE chats SET updated = ?, mode = ? WHERE id = ?", (t, mode, chat_id))

def save_file(chat_id: str, name: str, text: str):
    with closing(db()) as conn, conn:
        conn.execute(
            "INSERT INTO files (chat_id, name, text, created) VALUES (?, ?, ?, ?)",
            (chat_id, name, text, now()),
        )

def get_file(chat_id: str):
    """Returns (name, text) of the newest file attached to this chat, or None."""
    with closing(db()) as conn:
        row = conn.execute(
            "SELECT name, text FROM files WHERE chat_id = ? ORDER BY id DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
    return (row["name"], row["text"]) if row else None

init_db()
print("Syntrix rename build v14 loaded")

# --- Reading uploaded files ---
def extract_text(name: str, data: bytes):
    """Pulls plain text out of an uploaded file.
    Returns (text, was_cut). Raises HTTPException with a friendly message on problems."""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    if ext == "pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise HTTPException(status_code=400, detail="That PDF is password-protected, so I cannot read it.")
            parts, total = [], 0
            for page in reader.pages:
                piece = page.extract_text() or ""
                parts.append(piece)
                total += len(piece)
                if total > MAX_FILE_CHARS:   # enough text, stop reading more pages
                    break
            text = "\n".join(parts)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="I could not open that PDF. It may be damaged.")
    elif ext in TEXT_EXTENSIONS:
        text = data.decode("utf-8", errors="replace")
    else:
        raise HTTPException(
            status_code=400,
            detail="That file type is not supported yet. Use a PDF or a text/code file (.txt, .py, .md, .json, .csv...).",
        )

    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="I found no readable text in that file. (A scanned PDF is just pictures, so it has no text to read.)",
        )
    if len(text) > MAX_FILE_CHARS:
        return text[:MAX_FILE_CHARS] + CUT_MARK, True
    return text, False

# --- Web search (free, no API key) ---
def to_keywords(message: str, max_words: int = 8) -> str:
    """Turn a natural-language question into a short keyword query."""
    words = re.findall(r"[A-Za-z0-9\.\-\+#]+", message)
    keywords = [w for w in words if w.lower() not in STOPWORDS]
    if not keywords:
        keywords = words
    return " ".join(keywords[:max_words])

def build_search_query(message: str, topic: str):
    """Turns a message into a keyword query. For follow-ups like "when was he
    born?", adds the topic of this chat's last standalone question.
    Returns (query, topic)."""
    own = to_keywords(message)
    lowered = message.lower().strip()
    words = {w.strip("?.,!") for w in lowered.split()}
    is_followup = bool(words & PRONOUNS) or lowered.startswith(FOLLOWUP_STARTS)

    if is_followup and topic:
        merged = []
        for w in (topic + " " + own).split():
            if w.lower() not in [m.lower() for m in merged]:
                merged.append(w)
        query = " ".join(merged[:8])
        print(f"Follow-up detected: '{message}' -> search '{query}'")
        return query, topic

    # Standalone question: remember its subject (capitalized words if any)
    if not is_followup:
        caps = [w for w in own.split() if w[:1].isupper()]
        topic = " ".join(caps[:4]) if caps else " ".join(own.split()[:4])
    return own, topic

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

# --- Request model and answer preparation ---
class ChatIn(BaseModel):
    message: str
    mode: str = "code"
    chat_id: Optional[str] = None
    attachment_name: Optional[str] = None   # set only on the message that attaches a file
    attachment_text: Optional[str] = None
    length: str = "normal"                  # Settings: "short", "normal" or "long"
    use_web: bool = True                    # Settings: web search on/off (General mode)

def prepare(payload: ChatIn, history, topic: str, file=None):
    """Everything that happens BEFORE the model starts writing: picks the model,
    gathers RAG / web / file context, builds the messages list.
    file is (name, text) or None.
    Returns (model, messages, options, warning_text, sources_text, topic)."""
    sources_text = ""
    warning_text = ""
    length = payload.length if payload.length in LENGTH_SCALE else "normal"

    if payload.mode == "auto":
        context = retrieve_context(payload.message)
        system = AUTO_PROMPT + f"\n\n--- Reference material ---\n{context}\n--- End reference ---"
        model = AUTO_MODEL
        options = {"num_predict": scaled(AUTO_MAX_TOKENS, length)}
    elif payload.mode == "general":
        system = GENERAL_PROMPT
        if not file:
            if payload.use_web:
                # No file attached and search is on: search the web as usual
                query, topic = build_search_query(payload.message, topic)
                results = web_search(query)
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
            else:
                # Web search switched off in Settings
                system += SEARCH_OFF_NOTE
        model = GENERAL_MODEL
        options = {"temperature": 0.2, "num_ctx": 4096,
                   "num_predict": scaled(GENERAL_MAX_TOKENS, length)}
    else:
        system = CODE_PROMPT
        model = CODE_MODEL
        options = {"num_predict": scaled(CODE_MAX_TOKENS, length)}

    if file:
        fname, ftext = file
        system += (
            f"\n\nThe user attached a file named '{fname}'. Use its contents to answer. "
            "Base your answer on the file, and if the file does not contain the answer, say so.\n"
            f"--- Attached file: {fname} ---\n{ftext}\n--- End attached file ---"
        )
        options["num_ctx"] = FILE_CTX   # bigger memory so the file fits

    system += LENGTH_HINTS[length]

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": payload.message})
    return model, messages, options, warning_text, sources_text, topic

def attach_if_new(chat_id: str, payload: ChatIn) -> str:
    """If this message carries a newly attached file, saves it for the chat.
    Returns the text the chat window should show for the user's message."""
    if payload.attachment_text and payload.attachment_name:
        name = payload.attachment_name[:120]
        save_file(chat_id, name, payload.attachment_text[: MAX_FILE_CHARS + len(CUT_MARK)])
        return f"📎 {name}\n{payload.message}"
    return payload.message

# --- Pages and history API ---
@app.get("/")
def home():
    return {"message": "Hello from Syntrix"}

@app.get("/chat")
def chat_page():
    return FileResponse("static/chat.html")

@app.post("/chat/clear")
def clear_chat():
    # Kept so older pages do not break. The page now starts a new saved chat instead.
    return {"status": "ok"}

@app.post("/api/upload")
def upload_file(file: UploadFile = File(...)):
    """Receives a file from the page, reads the text out of it, and sends the text
    back. The text is saved with the chat later, when the user sends their message."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="That file is too big. The limit is 5 MB.")
    name = file.filename or "file"
    text, was_cut = extract_text(name, data)
    print(f"Upload: '{name}' -> {len(text)} characters (cut: {was_cut})")
    return {"name": name, "text": text, "was_cut": was_cut, "chars": len(text)}

@app.get("/api/chats")
def list_chats():
    with closing(db()) as conn:
        rows = conn.execute(
            "SELECT id, title, mode, updated FROM chats ORDER BY updated DESC LIMIT 200"
        ).fetchall()
    return [dict(r) for r in rows]

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    with closing(db()) as conn:
        chat = conn.execute(
            "SELECT id, title, mode FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        rows = conn.execute(
            "SELECT role, display FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)
        ).fetchall()
    return {
        "id": chat["id"],
        "title": chat["title"],
        "mode": chat["mode"],
        "messages": [{"role": r["role"], "text": r["display"]} for r in rows],
    }

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    with closing(db()) as conn, conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM files WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    return {"status": "deleted"}

class RenameIn(BaseModel):
    title: str

@app.patch("/api/chats/{chat_id}")
def rename_chat(chat_id: str, body: RenameIn):
    """Sidebar > pencil icon: gives a chat a new name."""
    title = " ".join(body.title.split())[:60]
    if not title:
        raise HTTPException(status_code=400, detail="The name cannot be empty.")
    with closing(db()) as conn, conn:
        cur = conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Chat not found")
    return {"id": chat_id, "title": title}

@app.delete("/api/chats")
def delete_all_chats():
    """Settings > Clear all chats: wipes every chat, message and saved file."""
    with closing(db()) as conn, conn:
        count = conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM files")
        conn.execute("DELETE FROM chats")
    print(f"Cleared all chats ({count} deleted)")
    return {"status": "cleared", "deleted": count}

# --- Chat endpoints ---
# Non-streaming version (waits for the full answer). Handy for testing in /docs.
@app.post("/chat")
def chat_reply(payload: ChatIn):
    chat_id = get_or_create_chat(payload.chat_id, payload.message, payload.mode)
    try:
        user_display = attach_if_new(chat_id, payload)
        history = load_context(chat_id)
        model, messages, options, warning_text, sources_text, topic = prepare(
            payload, history, get_topic(chat_id), get_file(chat_id)
        )
        set_topic(chat_id, topic)
        add_message(chat_id, "user", payload.message, user_display, payload.mode)

        resp = ollama.chat(
            model=model, messages=messages, options=options, keep_alive=KEEP_ALIVE
        )
        reply = resp["message"]["content"]
        cut = CUT_OFF_NOTE if getattr(resp, "done_reason", None) == "length" else ""
        shown = warning_text + reply + cut + sources_text
        add_message(chat_id, "assistant", reply, shown, payload.mode)
        return {"reply": shown, "chat_id": chat_id}
    except Exception as e:
        return {"reply": f"⚠️ AI error: {e}", "chat_id": chat_id}

# Streaming version: sends the answer piece by piece as the model writes it.
# The chat id travels back in the X-Chat-Id response header.
@app.post("/chat/stream")
def chat_stream(payload: ChatIn):
    chat_id = get_or_create_chat(payload.chat_id, payload.message, payload.mode)
    user_display = attach_if_new(chat_id, payload)
    history = load_context(chat_id)
    topic = get_topic(chat_id)
    file = get_file(chat_id)

    def generate():
        shown = ""
        reply = ""

        def emit(text):
            nonlocal shown
            shown += text
            return text

        try:
            add_message(chat_id, "user", payload.message, user_display, payload.mode)
            model, messages, options, warning_text, sources_text, new_topic = prepare(
                payload, history, topic, file
            )
            set_topic(chat_id, new_topic)

            if warning_text:
                yield emit(warning_text)

            cut_off = False
            for chunk in ollama.chat(
                model=model, messages=messages, options=options,
                stream=True, keep_alive=KEEP_ALIVE,
            ):
                piece = chunk["message"]["content"]
                if piece:
                    reply += piece
                    yield emit(piece)
                if getattr(chunk, "done_reason", None) == "length":
                    cut_off = True

            if cut_off:
                yield emit(CUT_OFF_NOTE)
            if sources_text:
                yield emit(sources_text)
        except Exception as e:
            yield emit(f"\n\n⚠️ AI error: {e}")
        finally:
            # Also runs if the browser disconnects, so partial answers are kept.
            if shown:
                add_message(chat_id, "assistant", reply, shown, payload.mode)

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Chat-Id": chat_id,
        },
    )