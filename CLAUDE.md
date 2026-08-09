# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Retrieval-Augmented Generation (RAG) Q&A chatbot** powered by Claude AI. It allows users to upload PDF documents and ask questions about their contents. The application uses:
- **LLM:** Claude Sonnet 4.5 (via LangChain)
- **Embeddings:** HuggingFace sentence-transformers (all-MiniLM-L6-v2, runs locally)
- **Vector Database:** Chroma (in-memory)
- **UI:** Gradio web interface
- **PDF Processing:** PyPDFLoader

## Quick Start

### Setup
```bash
# Install dependencies (requires Python 3.13+)
uv sync

# Create .env file with your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### Run the Application
```bash
# Launch the Gradio web UI (default: http://localhost:7860)
uv run qabot.py
```

The application creates an interactive interface where users can:
1. Upload a PDF file
2. Enter a question about the PDF content
3. Receive an answer based on the document

## Architecture

The RAG pipeline in `qabot.py` follows this flow:

1. **Document Loading** (`document_loader`)
   - Extracts text from uploaded PDF files using PyPDFLoader
   - Handles layout, fonts, and positioning encoded in PDFs

2. **Text Chunking** (`text_splitter`)
   - Splits documents into 1000-character chunks with 50-character overlap
   - Prevents LLM/embedding model token limits while maintaining context

3. **Embedding** (`get_embedding_model`)
   - Uses HuggingFace `all-MiniLM-L6-v2` model (~90MB, cached locally after first run)
   - Converts text chunks into numeric vectors for semantic search
   - No API keys required - runs entirely on machine

4. **Vector Storage** (`vector_database`)
   - Chroma stores chunk embeddings in memory
   - Enables semantic search by meaning rather than keyword matching

5. **Retrieval + Generation** (`retriever_qa`)
   - LangChain chain combines retriever with Claude LLM
   - Retriever finds most relevant chunks for the user's question
   - Prompt instructs Claude to answer only using retrieved context
   - Returns "I don't know" if answer isn't in the document

6. **User Interface** (`rag_application`)
   - Gradio Interface wraps the RAG function
   - File input limited to single PDF (.pdf only)
   - Text input for questions (2-line placeholder)
   - Text output for answers

### LLM Configuration
- **Model:** claude-sonnet-4-5
- **Temperature:** 0.5 (balanced between factual and creative)
- **Max tokens:** 256 (limits answer length)
- **API Key:** Loaded from `ANTHROPIC_API_KEY` environment variable (never hardcoded)

## Key Files

- **qabot.py** (147 lines) - Complete application; all functions are self-contained and in a single file
- **pyproject.toml** - Project metadata and dependencies (uv-based)
- **.env** - Environment variables (API key goes here)
- **.python-version** - Specifies Python 3.13 requirement
- **uv.lock** - Locked dependency versions for reproducibility

## Development Tasks

### Modify the RAG Pipeline
Edit `qabot.py` functions directly:
- `text_splitter()` - Change chunk_size/overlap if document context is insufficient
- `get_embedding_model()` - Swap embedding model (check HuggingFace for alternatives)
- `retriever_qa()` - Modify the prompt template for different instruction style
- `get_llm()` - Adjust temperature or max_tokens; switch Claude model version

### Modify the UI
Update the `gr.Interface()` call to:
- Add/remove input fields
- Change file type restrictions
- Modify labels and placeholders
- Adjust output formatting

### Update Dependencies
```bash
# Add a new dependency
uv add package-name

# Update all dependencies
uv sync --upgrade

# Check for updates
uv pip list --outdated
```

### Test Changes Locally
```bash
# Run the app with live reloading
uv run qabot.py
# Then visit http://localhost:7860
```

## Environment Setup

### Required
- Python 3.13+
- ANTHROPIC_API_KEY in `.env` file

### Optional
- Modify .env for other configurations (none currently)

## Notes

- **No build step:** Application runs directly without compilation or packaging
- **No test framework:** Current project has no automated tests
- **No linting:** No configured linters (consider adding `ruff`, `black`, or `mypy` if expanding)
- **Single file architecture:** All logic lives in `qabot.py` for simplicity
- **Embedding model caching:** First run downloads ~90MB model; subsequent runs use cached version from `~/.cache/`
- **Vector DB persistence:** Chroma DB is in-memory and lost when app restarts (add persistence if needed)

## Common Modifications

**To use a different Claude model:**
```python
llm = ChatAnthropic(model="claude-opus-4-7", ...)  # or claude-haiku-4-5-20251001
```

**To persist vector database between runs:**
```python
# In vector_database(), add persist_directory parameter
vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db")
# And in retriever(), use the existing db if it exists
```

**To support multiple PDF uploads:**
Modify `gr.File(file_count="multiple")` and adjust `retriever_qa()` to handle file list.

**To change retrieval behavior:**
Adjust `vectordb.as_retriever(search_kwargs={"k": 3})` to retrieve more/fewer chunks (default is typically 4).
