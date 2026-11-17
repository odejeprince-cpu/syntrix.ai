import json

with open("data/amck.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Success! Loaded {len(data)} automotive Q&A pairs from local file.")