# doc-schema-extractor

Template-guided PDF/XLSX extraction pipeline for recurring supplier documents, with LangSmith tracing, Streamlit chat UI, and rotating file-based debug logging.

## Supported LLM Backends
- **Ollama** (local, default): `gemma4:e4b-it-qat`, `qwen3.5:2b`, `gemma4:e2b`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o4-mini`

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/anbilly19/doc-schema-extractor
cd doc-schema-extractor
uv sync
cp .env.example .env
```

## Logging

The app writes rotating debug logs to `./logs/doc_schema_extractor.log` by default.

Config in `.env`:

```bash
LOG_LEVEL=DEBUG
LOG_DIR=./logs
LOG_FILE=doc_schema_extractor.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
LOG_RAW_TEXT_PREVIEW_CHARS=2000
```

Logged events include:
- document intake and file type
- template store load/save/list/delete
- template match candidates + scores
- rule engine extraction per field
- validator failures
- LLM backend calls and parse failures
- Streamlit chat questions
- exception stack traces

Raw document text is truncated to `LOG_RAW_TEXT_PREVIEW_CHARS` to reduce leakage and log bloat.

## LangSmith

Set these in `.env` to enable tracing:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=doc-schema-extractor
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

## Run

```bash
uv sync
uv run dse extract path/to/invoice.pdf
uv run streamlit run src/doc_schema_extractor/streamlit_app.py
```

## License

MIT
