from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI


def _chunk_markdown(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Normalize whitespace a bit (keeps content readable, helps embedding consistency)
        p = re.sub(r"\s+", " ", p)
        chunks.append(p)
    return chunks


def _load_kb_chunks(kb_dir: str) -> list[dict]:
    chunks: list[dict] = []
    for name in sorted(os.listdir(kb_dir)):
        if not name.lower().endswith(".md"):
            continue
        path = os.path.join(kb_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        for i, chunk_text in enumerate(_chunk_markdown(content)):
            chunks.append(
                {
                    "id": f"{name}:{i}",
                    "source": name,
                    "text": chunk_text,
                }
            )
    return chunks


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build embeddings index for kb/*.md")
    parser.add_argument("--kb-dir", default="kb", help="Path to knowledge base directory")
    parser.add_argument("--out", default="kb_index.json", help="Output index JSON file")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embeddings model",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set (put it in .env or env vars).")

    kb_dir = os.path.abspath(args.kb_dir)
    if not os.path.isdir(kb_dir):
        raise SystemExit(f"KB dir not found: {kb_dir}")

    chunks = _load_kb_chunks(kb_dir)
    if not chunks:
        raise SystemExit("No KB chunks found. Ensure kb/ contains .md files.")

    client = OpenAI(api_key=api_key)
    texts = [c["text"] for c in chunks]

    embeddings: list[list[float]] = []
    for batch in _batched(texts, args.batch_size):
        resp = client.embeddings.create(model=args.model, input=batch)
        # Response preserves input order
        for item in resp.data:
            embeddings.append(item.embedding)

    if len(embeddings) != len(chunks):
        raise SystemExit("Embedding count mismatch. Aborting.")

    for c, e in zip(chunks, embeddings, strict=True):
        c["embedding"] = e

    index = {
        "version": 1,
        "embedding_model": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kb_dir": os.path.relpath(kb_dir, os.getcwd()),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    print(f"Wrote {len(chunks)} chunks to: {out_path}")
    print(f"Embedding model: {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
