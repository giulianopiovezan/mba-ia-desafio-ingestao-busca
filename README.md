# MBA AI Engineering Challenge — PDF RAG

A **Retrieval-Augmented Generation (RAG)** pipeline that ingests a PDF, stores vector embeddings in PostgreSQL (PGVector), and answers questions via a CLI chatbot using only document context.

## Stack

- Python, LangChain
- OpenAI embeddings (`text-embedding-3-small`) + ChatOpenAI
- PostgreSQL + PGVector (Docker)
- PyPDF for document loading

## Prerequisites

- Python 3.11+ (recommended)
- Docker (for the database)
- OpenAI-compatible API key
- A PDF file (default: `document.pdf` at the project root)

## Quick start

### 1. Clone and enter the project

```bash
git clone <repository-url>
cd mba-ia-desafio-ingestao-busca
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Always activate the venv before running scripts. Use `pip install -r requirements.txt` (with `-r`), not `pip install requirements.txt`.

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=langchain_challenge
PDF_PATH=document.pdf
```

### 4. Start PostgreSQL with PGVector

```bash
docker compose up -d
```

Wait until the `postgres` service is healthy. The `bootstrap_vector_ext` service enables the `vector` extension automatically.

### 5. Ingest the PDF

From the project root, with the venv active:

```bash
python src/ingest.py
```

This loads the PDF, splits it into chunks (~1000 chars, 150 overlap), generates embeddings, and stores them in PGVector.

### 6. Run the chatbot

```bash
python src/chat.py
```

Type your question when prompted. Press `Ctrl+C` to exit.

## Project layout

```
.
├── docker-compose.yml    # PostgreSQL + PGVector
├── requirements.txt      # Pinned dependencies
├── .env.example          # Environment template
├── document.pdf          # Sample document (or set PDF_PATH)
└── src/
    ├── ingest.py         # PDF ingestion pipeline
    ├── search.py         # Semantic search + RAG prompt
    ├── chat.py           # Interactive CLI
    └── utils.py          # PGVector store + env validation
```

## How it works

| Script | Role |
|--------|------|
| `ingest.py` | Load PDF → split → embed → store in PostgreSQL |
| `search.py` | Retrieve top-k similar chunks and build the RAG prompt |
| `chat.py` | Ask questions; answers are grounded in retrieved context only |

If the answer is not in the document, the model is instructed to reply: *"Não tenho informações necessárias para responder sua pergunta."*

## FAQ / Troubleshooting

**`pip install -r requirements.txt` installs nothing**

The venv may not be active. Run `which python` — it should point to `venv/bin/python`. If not, run `source venv/bin/activate` and try again.

**`ModuleNotFoundError: No module named 'langchain_community'`**

Dependencies were not installed in the active environment. Activate the venv and run `pip install -r requirements.txt`.

**`Environment variable OPENAI_API_KEY is not set`**

Copy `.env.example` to `.env` and fill in all required values. Scripts load `.env` automatically via `python-dotenv`.

**`connection refused` or database errors**

1. Check Docker is running: `docker compose ps`
2. Start the database: `docker compose up -d`
3. Confirm `DATABASE_URL` matches docker-compose credentials (`postgres` / `postgres` / `rag` on port `5432`)

**`FileNotFoundError` when running ingest**

Set `PDF_PATH` in `.env` to a valid PDF path, or place the file at the project root as `document.pdf`.

**Chat returns empty or irrelevant answers**

Run `python src/ingest.py` first. Re-run ingestion after changing the PDF or collection name.

**Need to use a different OpenAI-compatible endpoint**

`src/utils.py` and `src/chat.py` configure a custom `base_url`. Update those values if you are not using the default API host.

**Port 5432 already in use**

Another PostgreSQL instance is running locally. Stop it or change the host port in `docker-compose.yml` and update `DATABASE_URL` accordingly.
