# doc-schema-extractor

Template-guided PDF/XLSX extraction pipeline for recurring supplier documents, with LangSmith tracing and a Streamlit chat UI.

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
# Edit .env with your keys
```

## LangSmith

Set these in `.env` to enable tracing:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=doc-schema-extractor
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Traced spans:
- `extraction_run` — full pipeline per document
- `template_match` — keyword fingerprint scoring
- `rule_engine_apply` — deterministic field extraction
- `validator_run` — confidence checks
- `llm_template_generation` — LLM fallback + template creation
- `chat_turn` — each user question in the Streamlit UI

## CLI Usage

```bash
uv run dse extract path/to/invoice.pdf
uv run dse extract path/to/invoice.pdf --backend openai --model gpt-4o-mini
uv run dse batch ./docs/ --output results.json
uv run dse templates list
uv run dse templates show redefine_meat_order_confirmation_v1
uv run dse templates delete <template_id> --yes
```

## Chat UI

```bash
uv run streamlit run src/doc_schema_extractor/streamlit_app.py
```

- Upload a PDF or XLSX
- Choose backend and model
- Run extraction (shows template ID, match score, LLM used)
- Inspect extracted JSON
- Chat with the document using the extracted data + raw text as context
- All runs and chat turns traced in LangSmith

## Architecture

```
doc input
  ↓
[TextExtractor]  pdfplumber (MIT) + openpyxl (MIT)
  ↓
[TemplateStore.match()]  rapidfuzz keyword score
  ├── HIT → [RuleEngine] → [Validator] → result  (0 LLM calls)
  └── MISS/FAIL → [LLMBackend] → save template → result
```

## License

MIT
