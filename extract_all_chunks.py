#!/usr/bin/env python3
"""
Extract ALL quality chunks from ChromaDB for full dataset generation.
Run on M1 (100.107.30.22) where ChromaDB lives.

Usage:
  cd ~/Projects/medexpert-admin
  venv/bin/python3 /tmp/extract_all_chunks.py
"""

import chromadb
import json
import hashlib
from collections import defaultdict

CHROMADB_PATH = "data/experts/oncologia/chromadb"
COLLECTION_NAME = "clinical_guidelines"
OUTPUT_FILE = "/tmp/all_quality_chunks.jsonl"

MIN_CHUNK_LENGTH = 80
SKIP_PATTERNS = [
    "available at: http", "personal fees", "advisory board",
    "copyright ©", "doi.org", "all rights reserved",
    "this guideline was developed", "go to patient version",
    "figure legend", "table of contents",
]

STRATA_KEYWORDS = {
    "farmacologia": {"society": ["PHARMA"], "category": ["farmacia"]},
    "tratamiento": {"kw": ["treatment", "therapy", "regimen", "chemotherapy", "radiation", "surgery", "systemic", "neoadjuvant", "adjuvant", "tratamiento", "quimioterapia", "radioterapia", "cirugía", "terapia"]},
    "diagnostico": {"kw": ["diagnosis", "staging", "tnm", "biomarker", "pathology", "classification", "molecular", "immunohistochemistry", "diagnóstico", "estadificación", "biopsia"]},
    "soporte": {"kw": ["supportive", "palliative", "toxicity management", "adverse event", "pain", "nausea", "neutropenia", "cuidado paliativo", "dolor", "soporte"]},
    "seguimiento": {"kw": ["follow-up", "surveillance", "monitoring", "survivorship", "recurrence", "seguimiento", "vigilancia", "supervivencia"]},
}


def is_quality(text):
    if len(text.strip()) < MIN_CHUNK_LENGTH:
        return False
    tl = text.lower()
    for p in SKIP_PATTERNS:
        if p in tl:
            return False
    alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha >= 0.4


def classify(meta, text):
    society = meta.get("society", "")
    category = meta.get("category", "")
    if society == "PHARMA" or category == "farmacia":
        return "farmacologia"
    combined = (meta.get("source", "") + " " + meta.get("section_path", "") + " " + text[:500]).lower()
    scores = {}
    for name, cfg in STRATA_KEYWORDS.items():
        if name == "farmacologia":
            continue
        score = sum(1 for kw in cfg.get("kw", []) if kw in combined)
        scores[name] = score
    best = max(scores, key=scores.get) if scores else "tratamiento"
    return best if scores.get(best, 0) >= 1 else "tratamiento"


def main():
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    col = client.get_collection(COLLECTION_NAME)
    total = col.count()
    print(f"Total chunks: {total}")

    # Dedup tracking
    seen_hashes = set()
    written = 0
    skipped_quality = 0
    skipped_dup = 0
    strata_counts = defaultdict(int)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        batch_size = 10000
        for offset in range(0, total, batch_size):
            r = col.get(include=["metadatas", "documents"], limit=batch_size, offset=offset)
            if not r["documents"]:
                break

            for doc, meta, cid in zip(r["documents"], r["metadatas"], r["ids"]):
                if not is_quality(doc):
                    skipped_quality += 1
                    continue

                h = hashlib.md5(doc[:150].encode()).hexdigest()
                if h in seen_hashes:
                    skipped_dup += 1
                    continue
                seen_hashes.add(h)

                stratum = classify(meta, doc)
                strata_counts[stratum] += 1

                chunk = {
                    "id": cid,
                    "text": doc,
                    "source": meta.get("source", ""),
                    "society": meta.get("society", ""),
                    "category": meta.get("category", ""),
                    "section_path": meta.get("section_path", ""),
                    "stratum": stratum,
                }
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                written += 1

            print(f"  Processed {min(offset + batch_size, total)}/{total} | Written: {written}")

    print(f"\n=== DONE ===")
    print(f"Written: {written}")
    print(f"Skipped (quality): {skipped_quality}")
    print(f"Skipped (dup): {skipped_dup}")
    print(f"Strata: {dict(strata_counts)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
