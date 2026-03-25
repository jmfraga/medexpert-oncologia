#!/usr/bin/env python3
"""
Fase 1 — Muestreo estratificado de ChromaDB para fine-tuning llama8b-onco.

Ejecutar en M1 (100.107.30.22) donde está la ChromaDB:
  cd /Users/juanma/Projects/medexpert-admin
  venv/bin/python3 /tmp/01_sample_chromadb.py

Output: sampled_chunks.json (~12,000 chunks representativos)
"""

import chromadb
import json
import random
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

# ── Config ──
CHROMADB_PATH = "data/experts/oncologia/chromadb"
COLLECTION_NAME = "clinical_guidelines"
OUTPUT_FILE = "sampled_chunks.json"
TARGET_TOTAL = 12000

# Estratificación por tipo de contenido
# Se clasifica por source name patterns + metadata
STRATA = {
    "tratamiento": {
        "target_pct": 0.30,
        "description": "Protocolos de tratamiento (quimio, radioterapia, cirugía, inmunoterapia)",
        "source_patterns": [
            "breast", "nscl", "sclc", "colon", "rectal", "ovarian", "melanoma",
            "prostate", "pancreatic", "gastric", "esophageal", "cervical", "uterine",
            "bladder", "kidney", "hepatobiliary", "thyroid", "head_and_neck",
            "testicular", "sarcoma", "brain", "cns",
        ],
        "section_keywords": [
            "treatment", "therapy", "regimen", "chemotherapy", "radiation",
            "surgery", "surgical", "systemic", "neoadjuvant", "adjuvant",
            "first-line", "second-line", "tratamiento", "quimioterapia",
            "radioterapia", "cirugía", "terapia",
        ],
    },
    "farmacologia": {
        "target_pct": 0.25,
        "description": "Farmacología oncológica (mecanismos, dosis, toxicidades)",
        "society_match": ["PHARMA"],
        "category_match": ["farmacia"],
        "section_keywords": [
            "dose", "dosing", "pharmacology", "toxicity", "adverse",
            "side effect", "drug", "dosis", "farmaco", "medicamento",
            "interaccion", "contraindicacion", "ficha técnica",
        ],
    },
    "diagnostico": {
        "target_pct": 0.20,
        "description": "Diagnóstico y estadificación (TNM, biomarcadores, patología)",
        "section_keywords": [
            "diagnosis", "staging", "tnm", "biomarker", "pathology",
            "classification", "grade", "histology", "molecular",
            "immunohistochemistry", "fish", "her2", "egfr", "alk",
            "braf", "msi", "brca", "pd-l1", "ki67",
            "diagnóstico", "estadificación", "biopsia", "biomarcador",
        ],
    },
    "soporte": {
        "target_pct": 0.15,
        "description": "Manejo de efectos adversos y supportive care",
        "source_patterns": [
            "supportive", "palliative", "antiemesis", "pain",
            "survivorship", "distress",
        ],
        "section_keywords": [
            "supportive", "palliative", "toxicity management", "adverse event",
            "antiemetic", "pain", "nausea", "neutropenia", "anemia",
            "mucositis", "neuropathy", "fatigue", "psychosocial",
            "cuidado paliativo", "dolor", "náusea", "soporte",
        ],
    },
    "seguimiento": {
        "target_pct": 0.10,
        "description": "Guías de seguimiento y supervivencia",
        "source_patterns": ["survivorship", "follow"],
        "section_keywords": [
            "follow-up", "surveillance", "monitoring", "survivorship",
            "recurrence", "screening", "seguimiento", "vigilancia",
            "supervivencia", "control", "monitoreo",
        ],
    },
}

# ── Filters: skip low-quality chunks ──
MIN_CHUNK_LENGTH = 80  # Skip very short chunks
SKIP_PATTERNS = [
    "available at: http",      # Bibliography URLs
    "personal fees",           # Conflict of interest
    "advisory board",          # COI disclosures
    "copyright ©",             # Copyright notices
    "doi.org",                 # DOI references
    "all rights reserved",
    "this guideline was developed",
    "go to patient version",
    "figure legend",
    "table of contents",
]


def classify_chunk(meta: dict, text: str) -> str:
    """Classify a chunk into a stratum based on metadata and content."""
    text_lower = text.lower()
    source = meta.get("source", "").lower()
    society = meta.get("society", "")
    category = meta.get("category", "")
    section = meta.get("section_path", "").lower()
    combined = source + " " + section + " " + text_lower[:500]

    # Priority 1: Farmacología (by metadata)
    if society == "PHARMA" or category == "farmacia":
        return "farmacologia"

    # Priority 2: Check section keywords for each stratum
    scores = {}
    for stratum_name, config in STRATA.items():
        score = 0
        for kw in config.get("section_keywords", []):
            if kw in combined:
                score += 1
        # Boost if source pattern matches
        for pat in config.get("source_patterns", []):
            if pat in source:
                score += 2
        scores[stratum_name] = score

    # Return highest scoring stratum (minimum score 1)
    best = max(scores, key=scores.get)
    if scores[best] >= 1:
        return best

    # Default: tratamiento (largest bucket, most general)
    return "tratamiento"


def is_quality_chunk(text: str) -> bool:
    """Filter out low-quality chunks (bibliography, COI, etc.)."""
    if len(text.strip()) < MIN_CHUNK_LENGTH:
        return False
    text_lower = text.lower()
    for pattern in SKIP_PATTERNS:
        if pattern in text_lower:
            return False
    # Skip chunks that are mostly references (numbers + dots)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return False
    return True


def deduplicate(chunks: list, sim_threshold: int = 100) -> list:
    """Remove near-duplicate chunks based on first N chars hash."""
    seen = set()
    unique = []
    for chunk in chunks:
        key = hashlib.md5(chunk["text"][:sim_threshold].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def main():
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    col = client.get_collection(COLLECTION_NAME)
    total = col.count()
    print("Total chunks:", total)

    # ── Load all chunks ──
    print("Loading chunks (this may take a few minutes)...")
    all_chunks = []
    batch_size = 10000
    for offset in range(0, total, batch_size):
        r = col.get(include=["metadatas", "documents"], limit=batch_size, offset=offset)
        if not r["documents"]:
            break
        for doc, meta, cid in zip(r["documents"], r["metadatas"], r["ids"]):
            all_chunks.append({"id": cid, "text": doc, "metadata": meta})
        print("  loaded", min(offset + batch_size, total), "/", total)

    print("Total loaded:", len(all_chunks))

    # ── Filter quality ──
    print("Filtering low-quality chunks...")
    quality_chunks = [c for c in all_chunks if is_quality_chunk(c["text"])]
    print("After quality filter:", len(quality_chunks), "(removed", len(all_chunks) - len(quality_chunks), ")")

    # ── Classify into strata ──
    print("Classifying into strata...")
    strata_buckets = defaultdict(list)
    for chunk in quality_chunks:
        stratum = classify_chunk(chunk["metadata"], chunk["text"])
        strata_buckets[stratum].append(chunk)

    print("\nClassification results:")
    for name, chunks in sorted(strata_buckets.items()):
        target = int(TARGET_TOTAL * STRATA[name]["target_pct"])
        print("  " + name + ": " + str(len(chunks)) + " available, target " + str(target))

    # ── Stratified sampling ──
    print("\nSampling...")
    sampled = []
    for name, config in STRATA.items():
        target = int(TARGET_TOTAL * config["target_pct"])
        bucket = strata_buckets[name]

        # Deduplicate within stratum
        bucket = deduplicate(bucket)

        # Ensure diversity: sample from different sources
        by_source = defaultdict(list)
        for c in bucket:
            by_source[c["metadata"].get("source", "unknown")].append(c)

        selected = []
        # Round-robin from sources to ensure diversity
        source_list = list(by_source.values())
        random.shuffle(source_list)
        idx = 0
        while len(selected) < target and idx < target * 10:
            source_chunks = source_list[idx % len(source_list)]
            chunk_idx = idx // len(source_list)
            if chunk_idx < len(source_chunks):
                selected.append(source_chunks[chunk_idx])
            idx += 1

        # If not enough, fill from remaining
        if len(selected) < target:
            remaining = [c for c in bucket if c not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[:target - len(selected)])

        sampled.extend(selected[:target])
        print("  " + name + ": sampled " + str(len(selected[:target])) + " / target " + str(target))

    # ── Final dedup across strata ──
    sampled = deduplicate(sampled, sim_threshold=150)
    print("\nFinal after cross-strata dedup:", len(sampled))

    # ── Prepare output ──
    output = {
        "version": "1.0",
        "total_source_chunks": total,
        "sampled_count": len(sampled),
        "strata_config": {k: {"target_pct": v["target_pct"], "description": v["description"]} for k, v in STRATA.items()},
        "chunks": []
    }

    for chunk in sampled:
        output["chunks"].append({
            "id": chunk["id"],
            "text": chunk["text"],
            "source": chunk["metadata"].get("source", ""),
            "society": chunk["metadata"].get("society", ""),
            "category": chunk["metadata"].get("category", ""),
            "section_path": chunk["metadata"].get("section_path", ""),
            "stratum": classify_chunk(chunk["metadata"], chunk["text"]),
        })

    # ── Save ──
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nSaved to", OUTPUT_FILE)
    print("File size:", round(Path(OUTPUT_FILE).stat().st_size / 1024 / 1024, 1), "MB")

    # ── Stats ──
    final_strata = defaultdict(int)
    final_sources = set()
    for c in output["chunks"]:
        final_strata[c["stratum"]] += 1
        final_sources.add(c["source"])

    print("\n=== FINAL DISTRIBUTION ===")
    for k, v in sorted(final_strata.items()):
        print("  " + k + ": " + str(v))
    print("Unique sources represented:", len(final_sources))


if __name__ == "__main__":
    random.seed(42)
    main()
