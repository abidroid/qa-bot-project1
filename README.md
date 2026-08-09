# RAG Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF and ask questions about its contents. Answers are grounded in the document — Claude only uses what it finds in your file, not its training data.

Built with **Claude AI**, **LangChain**, **Chroma**, and **Gradio**.

---

## How it works

```
PDF file
  └─► Extract text (PyPDFLoader)
        └─► Split into 1 000-char chunks (RecursiveCharacterTextSplitter)
              └─► Embed chunks as vectors (HuggingFace all-MiniLM-L6-v2)
                    └─► Store in Chroma vector DB (in-memory)
                          └─► User asks a question
                                └─► Retrieve top-4 similar chunks
                                      └─► Claude answers from those chunks only
```

Instead of guessing from memory, Claude is handed the relevant passages from your PDF before generating a response. If the answer isn't in the document, it says so.

---

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key ([get one here](https://console.anthropic.com/))

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd qa-bot-project1

# 2. Install dependencies
uv sync

# 3. Add your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

The HuggingFace embedding model (~90 MB) downloads automatically on first run and is cached locally — no HuggingFace account or API key needed.

---

## Run

```bash
uv run qabot.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

1. Upload a PDF file
2. Type your question
3. Get an answer grounded in the document

---

## Project structure

```
qa-bot-project1/
├── qabot.py          # Complete application (~150 lines)
├── pyproject.toml    # Dependencies (uv)
├── uv.lock           # Locked dependency versions
├── .python-version   # Python 3.13 pin
└── .env              # API key (create this yourself, not committed)
```

All logic lives in `qabot.py`. The functions follow the pipeline in order:

| Function | Role |
|---|---|
| `document_loader` | Extracts text from the PDF using PyPDFLoader |
| `text_splitter` | Cuts text into overlapping 1 000-char chunks |
| `get_embedding_model` | Loads the local HuggingFace sentence-transformer |
| `vector_database` | Embeds chunks and stores them in Chroma |
| `retriever` | Orchestrates ingestion and returns a retriever object |
| `get_llm` | Configures the Claude Sonnet 4.5 client |
| `format_docs` | Joins retrieved chunks into a single context string |
| `retriever_qa` | Runs the full RAG chain and returns Claude's answer |
| `rag_application` | Gradio interface — wraps everything into a web UI |

---

## Configuration

All tunable settings are in `qabot.py`:

| What | Where | Default |
|---|---|---|
| Claude model | `get_llm()` | `claude-sonnet-4-5` |
| Temperature | `get_llm()` | `0.5` |
| Max response tokens | `get_llm()` | `256` |
| Chunk size | `text_splitter()` | `1000` chars |
| Chunk overlap | `text_splitter()` | `50` chars |
| Chunks retrieved | `retriever()` | `4` (Chroma default) |
| Embedding model | `get_embedding_model()` | `all-MiniLM-L6-v2` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `gradio` | Web UI |
| `langchain-anthropic` | Claude integration |
| `langchain-chroma` | Chroma vector DB integration |
| `langchain-community` | PyPDFLoader |
| `langchain-core` | Prompt templates, LCEL chain primitives |
| `langchain-huggingface` | Local embedding model runner |
| `langchain-text-splitters` | Recursive text chunking |
| `pypdf` | PDF byte parsing |
| `python-dotenv` | Loads `.env` into environment |
| `sentence-transformers` | PyTorch sentence embedding models |

---

## Notes

- **Vector DB is in-memory** — Chroma resets when the app restarts. To persist it, pass `persist_directory="./chroma_db"` to `Chroma.from_documents()`.
- **No tests or linter configured** — consider adding `ruff` and `pytest` if the project grows.
- **Single PDF per session** — re-uploading a new file rebuilds the vector DB from scratch.
