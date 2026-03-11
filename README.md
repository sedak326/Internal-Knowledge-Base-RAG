# Multimodal RAG

Companies accumulate a lot of documentation. Finding answers in a 200-page manual is slow, so this tool turns PDF documents into a local knowledge base you can query in plain language. Originally built to handle internal company manuals, with the goal of eventually helping support teams answer customer questions faster.

---

## Setup

```bash
pip install pymupdf pdfplumber pillow requests langchain-community tqdm rich
```

You need [Ollama](https://ollama.com) running locally with these models:

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llava              # image summarization
ollama pull ministral-3:14b   # answer generation
```

---

## Usage

### 1. Extract from PDF

```bash
python extract_dual.py path/to/document.pdf \
  --out out \
  --doc-name "Your Document Name" \
  --ollama-url http://localhost:11434 \
  --ollama-model llava
```

Outputs:
- `out/indexable.jsonl`, all chunks ready for embedding
- `out/images/`, extracted images as PNGs
- `out/headings_by_page.json`, chapter/section map

### 2. Index

```bash
python index.py \
  --jsonl ./out/indexable.jsonl \
  --chroma-dir ./chroma_db \
  --collection docs \
  --embedding-model nomic-embed-text \
  --batch-size 10
```

### 3. Query

```bash
python run_RAG.py \
  --chroma-dir ./chroma_db \
  --collection docs \
  --embedding-model nomic-embed-text \
  --k 8 \
  --temperature 0.0
```

Interactive terminal session. Type a question, get an answer with source citations. `quit` or `exit` to stop.

---

## How It Works

### Text

Extracted with PyMuPDF. Headings are detected by font size (anything significantly larger than the page median). Structured numbering like `6.4.` is parsed to track chapters and sections. Chunks are capped at 4,000 characters and split at heading boundaries.

### Code

The extractor calculates a monospace font ratio per block. If 60%+ of characters use a monospace font (Courier, Consolas, Monaco, etc.), the block is classified as code. This is backed up by pattern matching for HTTP methods, JSON structure, and curl commands. Code is wrapped in backtick fences and kept inside the surrounding prose chunk so the explanation and the example stay together.

### Tables

Extracted with pdfplumber, which is more reliable than PyMuPDF for grid structures. Each table is converted to TSV and stored as its own chunk with full chapter/section/page metadata.

### Images

Images are extracted as PNGs and summarized by `llava`. When an image appears mid-section, the surrounding prose being processed at that point is prepended to the image description as `[SURROUNDING CONTEXT]...[END CONTEXT]`. This means the image embedding includes what the document was explaining when the image appeared, not just what the image looks like in isolation.

### Metadata

Every chunk, regardless of type, carries:

```
document_name, chapter, section_title, page_number, content_type, id
```

This is embedded into the chunk text at query time so the model can cite exact sources.

### Retrieval and Generation

Queries are embedded with `nomic-embed-text` and the top 8 chunks are retrieved by cosine similarity from Chroma. The LLM (`ministral-3:14b`, temperature 0.0) is prompted to answer only from the retrieved context, cite document name and page number, preserve code verbatim, and format output in markdown.

---

## What's Next

- Extend to support customer support teams answering questions from product documentation
- Filtered retrieval by document, chapter, or date
- A web UI for non-technical users
