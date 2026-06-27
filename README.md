# doc-schema-extractor

Template-guided PDF/XLSX extraction pipeline for recurring supplier documents.

**Two-phase approach:**
- **Phase 1 (MISS):** LLM extracts data AND generates reusable extraction rules → saved as template
- **Phase 2 (HIT):** Pure deterministic rule-based extraction — zero LLM calls

## Supported LLM Backends
- **Ollama** (local, default): `gemma4:e4b-it-qat`, `qwen3.5:2b`, `gemma4:e2b`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o4-mini`

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/anbilly19/doc-schema-extractor
cd doc-schema-extractor

# Create venv and install
uv sync

# Copy and configure env
cp .env.example .env
# Edit .env with your settings
```

## Usage

```bash
# Extract a single document
uv run dse extract path/to/invoice.pdf

# Extract with explicit backend
uv run dse extract path/to/invoice.pdf --backend openai --model gpt-4o-mini

# List all stored templates
uv run dse templates list

# Show a specific template
uv run dse templates show redefine_meat_order_confirmation_v1

# Reset/delete a template (forces LLM re-learn on next run)
uv run dse templates delete <template_id>

# Batch process a folder
uv run dse batch path/to/docs/ --output results.json
```

## Python API

```python
from doc_schema_extractor import Extractor
from doc_schema_extractor.backends import OllamaBackend, OpenAIBackend

# Use Ollama (local)
backend = OllamaBackend(model="gemma4:e4b-it-qat")
extractor = Extractor(backend=backend)

result = extractor.extract("invoice.pdf")
print(result.data)          # extracted fields
print(result.template_id)   # which template was used
print(result.llm_used)      # True if LLM was called (MISS), False on HIT
```

## Template Store

Templates are stored as versioned JSON in `templates/store.json` (configurable).
Each template contains:
- `fingerprint`: keywords + supplier hint for matching
- `extraction_rules`: regex/table rules per field
- `confidence_checks`: validation constraints
- `metadata`: creation date, hit count, last updated

## Architecture

```
doc/
  ↓
[TextExtractor] pdfplumber / openpyxl
  ↓
[TemplateStore.match()] rapidfuzz keyword fingerprint
  ├── HIT (score > 0.75)
  │     ↓
  │   [RuleEngine.apply()] regex + table parser
  │     ↓
  │   [Validator] pydantic + confidence checks
  │     ↓ pass         ↓ fail
  │   return result   → LLM fallback + template update
  │
  └── MISS
        ↓
      [LLMBackend] Ollama or OpenAI
        ↓  (structured output: data + extraction_rules)
      [TemplateStore.save()] new template
        ↓
      return result
```

## License

MIT
