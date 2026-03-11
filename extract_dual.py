#!/usr/bin/env python3
"""
extract_dual.py — Structured PDF extractor for RAG 

Architecture:
  - Text  → chunked as-is, preserving chapter/page/document metadata
  - Code  → kept as-is, embedded inside their surrounding prose chunk (NOT chunked separately, this preserves context)
  - Images → summarized via Ollama LLM, stored as text descriptions
  - Tables → extracted as TSV, kept with section context

Each chunk includes: document_name, chapter, section_title, page_number
so the RAG can tell the user exactly where to look.

Outputs (JSONL):
  out/indexable.jsonl   — all chunks ready for embedding
  out/images.jsonl      — image metadata
  out/images/           — extracted PNGs
  out/headings_by_page.json - which chapter each page belongs to

Install:
  pip install pymupdf pdfplumber pillow requests

Usage:
  python extract_dual.py your/data/*.pdf --out out --doc-name "PayPal Docs" --ollama-url http://localhost:11434 --ollama-model llava
"""

from __future__ import annotations

import re
import json
import base64
import argparse
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image




#These are monospac fonts (every character takes up the same space, indicative of code blocks)
MONO_HINTS = ("courier", "consolas", "menlo", "monaco", "dejavusansmono", "sourcecode", "code", "mono")

# Chapter/section number patterns like "1.", "1.2.", "1.2.3", "Chapter 1", "Section 2"
CHAPTER_PATTERN = re.compile(
    r"^(Chapter\s+\d+|Section\s+\d+|\d+(\.\d+)*\.?\s+\S)",
    re.IGNORECASE
)

MAX_SECTION_CHARS = 4000  # max chars per prose chunk before splitting

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_monospace_font(font_name: str) -> bool:
    """checks if fontname returned by extract_text_blocks_pymupdf
    is one of the mono hints we previously defined"""
    fn = (font_name or "").lower().replace(" ", "")
    return any(h in fn for h in MONO_HINTS)


def normalize_ws_keep_indent(text: str) -> str:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def is_heading(text: str, size: float, median_size: float) -> bool:
    """A span looks like a heading if it's larger than median and matches heading pattern."""
    if size < median_size * 1.3:
        return False
    if len(text.strip()) < 3 or len(text.strip()) > 120:
        return False
    if re.fullmatch(r"[\d\W]+", text.strip()):
        return False
    return True


def parse_chapter_from_heading(heading: str) -> str:
    """Extract top-level chapter from a heading string like '6.4. Postman API' → 'Chapter 6'."""
    m = re.match(r"^(\d+)\.", heading.strip())
    if m:
        return f"Chapter {m.group(1)}"
    m2 = re.match(r"^Chapter\s+(\d+)", heading.strip(), re.IGNORECASE)
    if m2:
        return f"Chapter {m2.group(1)}"
    return "Unknown Chapter"

@dataclass
class TextSpan:
    text: str
    size: float
    font: str
    flags: int


def extract_spans(page: fitz.Page) -> List[TextSpan]:
    spans = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for sp in line.get("spans", []):
                t = sp.get("text", "").strip()
                if t:
                    spans.append(TextSpan(
                        text=t,
                        size=float(sp.get("size") or 0),
                        font=str(sp.get("font") or ""),
                        flags=int(sp.get("flags") or 0),
                    ))
    return spans


def guess_headings_for_page(page: fitz.Page) -> List[str]:

    """We use pdfpblumber to detect tables. 
    It finds tables, converts them to TSV format, 
    and stores them with their page number. 
    Each table becomes its own chunk so it can be retrieved independently."""
    
    spans = extract_spans(page)
    if not spans:
        return []
    sizes = sorted(s.size for s in spans if s.size > 0)
    if not sizes:
        return []
    median = sizes[len(sizes) // 2]
    seen = set()
    out = []
    for s in spans:
        if is_heading(s.text, s.size, median):
            key = s.text.strip().lower()
            if key not in seen:
                seen.add(key)
                out.append(s.text.strip())
    return out

def extract_text_blocks_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text blocks from PDF. Each block preserves:
    - page_number
    - element_id
    - text (raw, with indentation preserved for code detection)
    - mono_ratio (fraction of monospace chars — used to detect code blocks)

    We are using PyMuPDF because it reads the metadata for every character like font, intendation
    and mono_ratio (what fraction of characters in the block used a monospace font like Courier or Consolas, 
    this is usually what codeblocks use) 
    This is the information that helps distinguish code from regular text and headers from body.
    """
    doc = fitz.open(str(pdf_path))
    out = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        for b_i, block in enumerate(page.get_text("dict").get("blocks", [])):
            if block.get("type") != 0:
                continue

            pieces = []
            mono_chars = 0
            total_chars = 0
            font_counts: Dict[str, int] = {}

            for line in block.get("lines", []):
                for sp in line.get("spans", []):
                    t = sp.get("text", "")
                    if not t:
                        continue
                    pieces.append(t)
                    font = str(sp.get("font") or "")
                    font_counts[font] = font_counts.get(font, 0) + len(t)
                    total_chars += len(t)
                    if is_monospace_font(font):
                        mono_chars += len(t)

            text = normalize_ws_keep_indent("".join(pieces))
            if not text.strip():
                continue

            mono_ratio = (mono_chars / total_chars) if total_chars else 0.0
            out.append({
                "page_number": page_index + 1,
                "element_id": f"p{page_index+1}_t{b_i}",
                "text": text,
                "mono_ratio": round(mono_ratio, 3),
            })

    doc.close()
    return out


def extract_tables_pdfplumber(pdf_path: Path) -> List[Dict[str, Any]]:
    out = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for t_i, table in enumerate(tables):
                if not table:
                    continue
                rows = [[(c or "").strip() for c in row] for row in table]
                rows = [r for r in rows if any(cell for cell in r)]
                if not rows:
                    continue
                tsv = "\n".join("\t".join(r).rstrip() for r in rows).strip()
                if tsv:
                    out.append({
                        "page_number": page_index + 1,
                        "element_id": f"p{page_index+1}_tbl{t_i}",
                        "text": tsv,
                        "rows": rows,
                    })
    return out


def extract_images_pymupdf(pdf_path: Path, out_dir: Path) -> List[Dict[str, Any]]:

    """Images are first extracted as PNG (./output_directory/images.json) 
    and then summarised later."""

    doc = fitz.open(str(pdf_path))
    img_dir = out_dir / "images"
    ensure_dir(img_dir)
    out = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_name = f"page_{page_index+1:03d}_img_{img_i:03d}.png"
                img_path = img_dir / img_name
                pix.save(str(img_path))
                pix = None
            except Exception:
                continue

            out.append({
                "page_number": page_index + 1,
                "element_id": f"p{page_index+1}_img{img_i}",
                "image_name": img_name,
                "image_path": str(img_path),
            })

    doc.close()
    return out


def summarize_image_ollama(image_path: str, ollama_url: str, model: str) -> str:
    """
    Send image to Ollama vision model (e.g. llava) for a description.
    Returns a plain-text summary of what the image shows.
    """
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": model,
            "prompt": (
                "You are analyzing a screenshot or diagram from a technical API documentation PDF. "
                "Describe what this image shows in detail: any UI elements, flowcharts, diagrams, "
                "tables, or code visible. Focus on what a developer would need to know from it. "
                "Be specific and thorough. Do not say 'the image shows' — just describe it directly."
            ),
            "images": [img_b64],
            "stream": False,
        }
        resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  [WARN] Image summarization failed for {image_path}: {e}")
        return ""


def block_is_code(text: str, mono_ratio: float) -> bool:
    """
    Decide if an entire text block should be treated as code.
    Uses monospace font ratio and content heuristics.
    """
    if mono_ratio >= 0.6:
        return True

    # HTTP endpoint pattern
    if re.match(r"^(GET|POST|PUT|DELETE|PATCH)\s+https?://", text.strip()):
        return True

    # JSON block
    stripped = text.strip()
    if (stripped.startswith("{") or stripped.startswith("[")) and (
        stripped.endswith("}") or stripped.endswith("]")
    ):
        return True

    # curl command
    if stripped.startswith("curl "):
        return True

    return False


def format_block_with_code(text: str, mono_ratio: float) -> str:
    """
    Return the block text formatted for the chunk.
    Code blocks are wrapped in ``` fences; prose is returned as-is.
    """
    if block_is_code(text, mono_ratio):
        return f"```\n{text}\n```"
    return text


# ---------------------------------------------------------------------------
# Main chunk builder
# ---------------------------------------------------------------------------

def build_docs(
    document_name: str,
    text_elements: List[Dict[str, Any]],
    table_elements: List[Dict[str, Any]],
    image_elements: List[Dict[str, Any]],
    headings_by_page: Dict[int, List[str]],
    ollama_url: str,
    ollama_model: str,
) -> List[Dict[str, Any]]:
    """
    Build indexable chunks preserving document_name, chapter, section, page.

    Strategy:
    - Text and code blocks are kept TOGETHER in prose chunks (code embedded as fenced blocks).
      This means the retrieval system always finds endpoints WITH their explanation.
    - Each chunk stays under MAX_SECTION_CHARS to keep context windows manageable.
    - When a new heading is detected, a new chunk starts.
    - Tables and images become their own separate chunks with the same metadata.
    """
    docs: List[Dict[str, Any]] = []

    # --- State ---
    current_section = "Introduction"
    current_chapter = "Chapter 1"
    chunk_buf: List[str] = []
    chunk_pages: List[int] = []
    chunk_counter = 0

    def flush_chunk() -> None:
        nonlocal chunk_buf, chunk_pages, chunk_counter
        if not chunk_buf:
            return
        text = "\n\n".join(chunk_buf).strip()
        if not text:
            chunk_buf = []
            chunk_pages = []
            return
        chunk_counter += 1
        page_start = min(chunk_pages) if chunk_pages else None
        page_end = max(chunk_pages) if chunk_pages else None
        page_ref = page_start if page_start == page_end else f"{page_start}-{page_end}"

        docs.append({
            "content_type": "text",
            "id": f"chunk_{chunk_counter:04d}",
            "document_name": document_name,
            "chapter": current_chapter,
            "section_title": current_section,
            "page_number": page_start,
            "page_range": page_ref,
            "text": (
                f"[Document: {document_name} | Chapter: {current_chapter} | "
                f"Section: {current_section} | Page: {page_ref}]\n\n{text}"
            ),
        })
        chunk_buf = []
        chunk_pages = []

    def elem_sort_key(e: Dict[str, Any]) -> tuple:
        return (int(e.get("page_number") or 0), str(e.get("element_id") or ""))

    # --- Process text elements in page order ---
    for e in sorted(text_elements, key=elem_sort_key):
        page = int(e.get("page_number") or 0)
        raw = (e.get("text") or "").strip()
        if not raw:
            continue

        mono_ratio = e.get("mono_ratio", 0.0)

        # Check if a new heading appears on this page
        page_headings = headings_by_page.get(page, [])
        if page_headings:
            new_section = page_headings[0]
            if new_section and new_section.strip() != current_section:
                flush_chunk()
                current_section = new_section.strip()
                current_chapter = parse_chapter_from_heading(current_section)

        # Format the block (wraps code in fences, leaves prose as-is)
        formatted = format_block_with_code(raw, mono_ratio)

        # If adding this block would make the chunk too large, flush first
        current_len = sum(len(b) for b in chunk_buf)
        if current_len + len(formatted) > MAX_SECTION_CHARS and chunk_buf:
            flush_chunk()

        chunk_buf.append(formatted)
        chunk_pages.append(page)

    flush_chunk()

    # --- Tables ---
    for t in sorted(table_elements, key=elem_sort_key):
        page = int(t.get("page_number") or 0)
        headings = headings_by_page.get(page, [])
        sec = headings[0] if headings else current_section
        chap = parse_chapter_from_heading(sec)

        docs.append({
            "content_type": "table",
            "id": t.get("element_id"),
            "document_name": document_name,
            "chapter": chap,
            "section_title": sec,
            "page_number": page,
            "page_range": page,
            "text": (
                f"[Document: {document_name} | Chapter: {chap} | "
                f"Section: {sec} | Page: {page}]\n\n[TABLE]\n{t.get('text', '')}"
            ),
        })

    # --- Images ---
    total_images = len(image_elements)
    for idx, im in enumerate(sorted(image_elements, key=elem_sort_key)):
        page = int(im.get("page_number") or 0)
        headings = headings_by_page.get(page, [])
        sec = headings[0] if headings else current_section
        chap = parse_chapter_from_heading(sec)
        img_path = im.get("image_path", "")
        img_name = im.get("image_name", "")

        print(f"  Processing image {idx+1}/{total_images}: {img_name}")
        print(f"    Summarizing with {ollama_model}...")

        description = summarize_image_ollama(img_path, ollama_url, ollama_model)
        if description:
            print(f"    Summary: {description[:80]}...")

        surrounding_context = "\n\n".join(chunk_buf).strip()

        body = f"[IMAGE: {img_name}]"
        if description:
            body += f"\n\n{description}"
        else:
            body += "\n\n[No description available — image summarization failed]"

        if surrounding_context:
            body = f"[SURROUNDING CONTEXT]\n{surrounding_context}\n\n[END CONTEXT]\n\n{body}"

        docs.append({
            "content_type": "image",
            "id": im.get("element_id"),
            "document_name": document_name,
            "chapter": chap,
            "section_title": sec,
            "page_number": page,
            "page_range": page,
            "image_name": img_name,
            "image_path": img_path,
            "text": (
                f"[Document: {document_name} | Chapter: {chap} | "
                f"Section: {sec} | Page: {page}]\n\n{body}"
            ),
        })

    return docs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract PDF content for RAG indexing")
    ap.add_argument("pdf", type=str, help="Path to PDF file or glob (e.g. docs/*.pdf)")
    ap.add_argument("--out", type=str, default="out", help="Output directory")
    ap.add_argument("--doc-name", type=str, default=None,
                    help="Human-readable document name (default: PDF filename)")
    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                    help="Ollama base URL (default: http://localhost:11434)")
    ap.add_argument("--ollama-model", type=str, default="llava",
                    help="Ollama vision model for image summarization (default: llava)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    document_name = args.doc_name or pdf_path.stem.replace("_", " ").replace("-", " ").title()
    print(f"\nDocument: {document_name}")
    print(f"PDF:      {pdf_path}")
    print(f"Output:   {out_dir}\n")

    print("Extracting text blocks...")
    text_elements = extract_text_blocks_pymupdf(pdf_path)
    print(f"  {len(text_elements)} text blocks")

    print("Extracting tables...")
    table_elements = extract_tables_pdfplumber(pdf_path)
    print(f"  {len(table_elements)} tables")

    print("Extracting images...")
    image_elements = extract_images_pymupdf(pdf_path, out_dir)
    print(f"  {len(image_elements)} images")

    print("Detecting headings...")
    doc = fitz.open(str(pdf_path))
    headings_by_page: Dict[int, List[str]] = {}
    for i in range(len(doc)):
        headings_by_page[i + 1] = guess_headings_for_page(doc[i])
    doc.close()

    print("\nBuilding chunks...")
    docs = build_docs(
        document_name=document_name,
        text_elements=text_elements,
        table_elements=table_elements,
        image_elements=image_elements,
        headings_by_page=headings_by_page,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
    )

    # Write outputs
    (out_dir / "headings_by_page.json").write_text(
        json.dumps(headings_by_page, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(out_dir / "indexable.jsonl", docs)
    write_jsonl(out_dir / "images.jsonl", image_elements)

    from collections import Counter
    counts = Counter(d["content_type"] for d in docs)
    print("\nDone!")
    print("Chunk counts:", dict(counts))
    print("Outputs:")
    print(f"  {out_dir / 'indexable.jsonl'}")
    print(f"  {out_dir / 'images.jsonl'}")
    print(f"  {out_dir / 'headings_by_page.json'}")
    print(f"  {out_dir / 'images/'}")


if __name__ == "__main__":
    main()