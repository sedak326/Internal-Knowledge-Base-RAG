#!/usr/bin/env python3
"""
index.py — Load indexable.jsonl and embed into a Chroma vector store

Usage:
  python index.py --jsonl ./out/indexable.jsonl --chroma-dir ./chroma_db --collection paypal_docs_v1
"""

import argparse
import json
import os
import time
from collections import Counter
from typing import List

from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata


def load_documents(jsonl_path: str) -> List[Document]:
    docs: List[Document] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = (item.get("text") or "").strip()
            if not text:
                continue
            metadata = {k: v for k, v in item.items() if k != "text"}
            metadata["content_type"] = item.get("content_type", "text")
            for key in list(metadata.keys()):
                if isinstance(metadata[key], (dict, list)):
                    metadata[key] = json.dumps(metadata[key])
            docs.append(Document(page_content=text, metadata=metadata))
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed and index chunks into Chroma")

    ap.add_argument("--jsonl", type=str, default="./out/indexable.jsonl",
                    help="Path to indexable.jsonl from extract_dual.py (default: ./out/indexable.jsonl)")
    ap.add_argument("--chroma-dir", type=str, default="./chroma_db",
                    help="Directory to persist the Chroma vector store (default: ./chroma_db)")
    ap.add_argument("--collection", type=str, default="docs",
                    help="Chroma collection name (default: docs)")

    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                    help="Ollama base URL (default: http://localhost:11434)")
    ap.add_argument("--embedding-model", type=str, default="nomic-embed-text",
                    help="Ollama embedding model (default: nomic-embed-text)")

    ap.add_argument("--batch-size", type=int, default=10,
                    help="Number of documents per embedding batch (default: 10)")

    args = ap.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading documents from: {args.jsonl}")
    docs = load_documents(args.jsonl)
    docs = filter_complex_metadata(docs)

    print(f"✓ Loaded {len(docs)} documents")
    print("Content types:", Counter(d.metadata.get("content_type") for d in docs))

    # ── Embeddings sanity check ───────────────────────────────────────────────
    print(f"\nConnecting to Ollama at {args.ollama_url}...")
    embeddings = OllamaEmbeddings(
        model=args.embedding_model,
        base_url=args.ollama_url,
    )
    test = embeddings.embed_query("test")
    print(f"✓ Embeddings working — vector dim: {len(test)}")

    # ── Index ─────────────────────────────────────────────────────────────────
    os.makedirs(args.chroma_dir, exist_ok=True)
    print(f"\nIndexing {len(docs)} documents in batches of {args.batch_size}...")
    print(f"Output: {args.chroma_dir}  (collection: {args.collection})\n")

    vectorstore = None
    start_time = time.time()

    with tqdm(total=len(docs), desc="Embedding") as pbar:
        for i in range(0, len(docs), args.batch_size):
            batch = docs[i : i + args.batch_size]
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=args.chroma_dir,
                        collection_name=args.collection,
                    )
                else:
                    vectorstore.add_documents(batch)
                pbar.update(len(batch))

                if (i + len(batch)) % 200 == 0:
                    vectorstore.persist()

            except Exception as e:
                import traceback
                print(f"\n⚠️  Batch {i // args.batch_size + 1} failed: {e}")
                traceback.print_exc()
                break

    if vectorstore is None:
        raise RuntimeError("Indexing failed — vectorstore was never created. Check errors above.")

    vectorstore.persist()
    elapsed = time.time() - start_time
    print(f"\n✓ Done in {elapsed/60:.1f} min ({len(docs)/elapsed:.1f} docs/sec)")
    print(f"✓ Chroma collection size: {vectorstore._collection.count()}")
    print(f"✓ Saved to: {args.chroma_dir}")


if __name__ == "__main__":
    main()