"""Load the bundled sample documents into RagFlowPro.

Run this after the stack is up so anyone can try the system on the included
sample data. For a fully local, no cost run:

    make db-up
    ollama serve &
    ollama pull nomic-embed-text
    EMBEDDING_PROVIDER=ollama python scripts/load_sample_data.py

Each file is chunked, embedded with the configured provider, and stored in the
pgvector database, exactly as an upload through the UI would be.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.worker.tasks import process_document  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def main() -> None:
    files = sorted(SAMPLE_DIR.glob("*.txt"))
    if not files:
        print(f"No .txt sample files found in {SAMPLE_DIR}")
        return
    print(f"Loading {len(files)} sample documents from {SAMPLE_DIR.name}/")
    for path in files:
        result = process_document(str(path), path.name)
        print(f"  {path.name}: {result}")
    print("Done. Open the UI and ask a question, see sample_data/README.md for examples.")


if __name__ == "__main__":
    main()
