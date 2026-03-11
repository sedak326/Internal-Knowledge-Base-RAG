#!/usr/bin/env python3
"""
run_RAG.py — Interactive terminal UI for the RAG query pipeline

Usage:
  python run_RAG.py --chroma-dir ./chroma_db --collection paypal_docs_v1
"""

import argparse

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.rule import Rule
    RICH = True
except ImportError:
    RICH = False


console = Console() if RICH else None


def print_answer(question: str, answer: str) -> None:
    if RICH:
        console.print()
        console.print(Panel(
            Markdown(answer),
            title=f"[bold blue] {question}[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        ))
    else:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")
        print(answer)
        print()


def print_status(msg: str, style: str = "dim") -> None:
    if RICH:
        console.print(f"[{style}]{msg}[/{style}]")
    else:
        print(msg)


def print_header(chroma_dir: str, collection: str, llm_model: str, embedding_model: str) -> None:
    if RICH:
        console.print()
        console.print(Rule("[bold blue]🔍 RAG Documentation Assistant[/bold blue]"))
        console.print(f"  [dim]Vector store:[/dim]    {chroma_dir}  [dim](collection: {collection})[/dim]")
        console.print(f"  [dim]LLM:[/dim]            {llm_model}")
        console.print(f"  [dim]Embeddings:[/dim]     {embedding_model}")
        console.print(f"  [dim]Commands:[/dim]       [bold]quit[/bold] or [bold]exit[/bold] to stop, [bold]clear[/bold] to clear screen")
        console.print(Rule())
        console.print()
    else:
        print("\n" + "="*60)
        print("  RAG Documentation Assistant")
        print(f"  Store: {chroma_dir} | Collection: {collection}")
        print(f"  LLM: {llm_model} | Embeddings: {embedding_model}")
        print("  Type 'quit' or 'exit' to stop, 'clear' to clear screen")
        print("="*60 + "\n")


def ask_question(question: str, vectorstore, llm, k: int = 8) -> str:
    retrieved = vectorstore.similarity_search(question, k=k)

    if not retrieved:
        return "I couldn't find any relevant information in the documentation for that question."

    context = "\n\n---\n\n".join(d.page_content for d in retrieved)

    prompt = f"""You are a technical documentation assistant.

STRICT RULES:
1. Answer ONLY using the context provided below. Do not use any prior knowledge.
2. If the answer is not in the context, say: "I don't have enough information in the documentation to answer that."
3. Always tell the user the document name and page number where they can find the information.
   Use the location headers in the context (e.g. "Document: PayPal Docs | Page: 60") to cite sources.
   Say something like: "You can find this on **page 60** of the **PayPal Docs**."
   Never say "SOURCE 1" or "SOURCE 2" — always use the actual document name and page.
4. Copy all endpoints, curl commands, JSON, and headers verbatim — never modify URLs or flags.
5. Format your answer clearly:
   - Use **bold** for important terms, endpoints, and field names.
   - Use numbered steps for how-to instructions.
   - Use code blocks for any commands, JSON, or API calls.
   - Use bullet points for lists of options or parameters.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    return llm.invoke(prompt)


def run_ui(vectorstore, llm, k: int) -> None:
    while True:
        try:
            if RICH:
                question = Prompt.ask("[bold blue]Ask a question[/bold blue]").strip()
            else:
                question = input("Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print_status("\n\nGoodbye!", "green")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print_status("\nGoodbye!", "green")
            break

        if question.lower() == "clear":
            if RICH:
                console.clear()
            else:
                print("\033c", end="")
            continue

        print_status("\n⏳ Searching documentation...", "dim")

        try:
            answer = ask_question(question, vectorstore, llm, k=k)
            print_answer(question, answer)
        except Exception as e:
            if RICH:
                console.print(f"[red]❌ Error: {e}[/red]")
            else:
                print(f"Error: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive RAG terminal UI")

    ap.add_argument("--chroma-dir", type=str, default="./chroma_db",
                    help="Path to the persisted Chroma directory (default: ./chroma_db)")
    ap.add_argument("--collection", type=str, default="docs",
                    help="Chroma collection name (default: docs)")

    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                    help="Ollama base URL (default: http://localhost:11434)")
    ap.add_argument("--embedding-model", type=str, default="nomic-embed-text",
                    help="Ollama embedding model (default: nomic-embed-text)")
    LLM_MODEL = "ministral-3:14b"

    ap.add_argument("--k", type=int, default=8,
                    help="Number of chunks to retrieve (default: 8)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="LLM temperature (default: 0.0)")

    args = ap.parse_args()

    if not RICH:
        print("Tip: install 'rich' for a nicer UI:  pip install rich")

    print_status("Loading vector store...", "dim")
    embeddings = OllamaEmbeddings(
        model=args.embedding_model,
        base_url=args.ollama_url,
    )
    vectorstore = Chroma(
        persist_directory=args.chroma_dir,
        embedding_function=embeddings,
        collection_name=args.collection,
    )
    print_status(f"Collection size: {vectorstore._collection.count()}", "dim")

    print_status(f"Loading LLM: {LLM_MODEL}...", "dim")
    llm = Ollama(
        model=LLM_MODEL,
        base_url=args.ollama_url,
        temperature=args.temperature,
    )

    print_header(args.chroma_dir, args.collection, LLM_MODEL, args.embedding_model)
    run_ui(vectorstore, llm, k=args.k)


if __name__ == "__main__":
    main()