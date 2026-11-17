import json
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading data...")
with open("data/amck.json", "r", encoding="utf-8") as f:
    pairs = json.load(f)

print(f"Loaded {len(pairs)} pairs. Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Building ChromaDB index...")
client = chromadb.PersistentClient(path="chroma_db")

# Delete the collection if it exists so reruns don't duplicate data
try:
    client.delete_collection("amck")
except Exception:
    pass

col = client.create_collection("amck")

texts = [f"Q: {p['question']}\nA: {p['answer']}" for p in pairs]
embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

col.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=[{"question": p["question"]} for p in pairs],
    ids=[f"amck-{i}" for i in range(len(pairs))],
)

print(f"✅ Indexed {len(pairs)} Q&A pairs into ChromaDB at ./chroma_db")