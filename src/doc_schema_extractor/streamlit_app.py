"""Streamlit chat UI for doc-schema-extractor."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

from doc_schema_extractor.logging_utils import get_logger

load_dotenv()
logger = get_logger("streamlit_app")

OLLAMA_MODELS = ["gemma4:e4b-it-qat", "qwen3.5:2b", "gemma4:e2b"]
OPENAI_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "gpt-4o",
    "o4-mini",
    "gpt-5",
    "gpt-5-mini",
]


def _default_backend() -> str:
    return os.getenv("LLM_BACKEND", "openai").lower()


def _answer_ollama(prompt: str, model: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "300"))
    payload = {
        "model": model, "prompt": prompt, "stream": True,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    logger.debug("Chat->Ollama (streaming) model=%s prompt_chars=%s", model, len(prompt))
    chunks: list[str] = []
    t = httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=5.0)
    with httpx.Client(timeout=t) as client:
        with client.stream("POST", f"{base_url}/api/generate", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunks.append(chunk.get("response", ""))
                if chunk.get("done"):
                    break
    return "".join(chunks) or "No response."


def _answer_openai(prompt: str, model: str) -> str:
    from openai import OpenAI
    logger.debug("Chat->OpenAI model=%s prompt_chars=%s", model, len(prompt))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model, temperature=0.1,
        messages=[
            {"role": "system", "content": "You answer questions about extracted supply chain documents."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or "No response."


def answer_question(question: str, extraction_result: dict, backend_name: str, model: str) -> str:
    from doc_schema_extractor.tracing import trace_chat_turn

    logger.info("Chat question backend=%s model=%s question=%s", backend_name, model, question)
    data_str = json.dumps(extraction_result.get("data", {}), ensure_ascii=False, indent=2)
    raw_text = (extraction_result.get("raw_text") or "")[:8000]
    prompt = (
        "You are a supply chain document assistant. Answer questions based on the extracted data "
        "and document text below. Be concise. If something is not present, say so.\n\n"
        f"=== Extracted fields ===\n{data_str}\n\n"
        f"=== Raw document text (preview) ===\n{raw_text}\n\n"
        f"Question: {question}"
    )
    if backend_name == "openai":
        answer = _answer_openai(prompt, model)
    else:
        answer = _answer_ollama(prompt, model)

    trace_chat_turn(
        question=question,
        template_id=extraction_result.get("template_id"),
        backend=backend_name,
        model=model,
        answer=answer,
        llm_used=extraction_result.get("llm_used", False),
    )
    return answer


def _render_score_dashboard() -> None:
    from doc_schema_extractor.audit_log import AuditLog
    audit = AuditLog()
    records = audit.read_all()
    if not records:
        st.info("No extraction runs recorded yet.")
        return

    st.subheader("Cross-document extraction scores")
    rows = []
    for r in records:
        rows.append({
            "File": r.get("file", ""),
            "Template": r.get("template_id") or "—",
            "Score": f"{r.get('match_score', 0):.3f}",
            "Result": r.get("result", ""),
            "LLM": "✓" if r.get("llm_used") else "",
            "Valid": "✓" if r.get("validation_passed") else "✗",
            "Fields": r.get("field_count", 0),
            "ms": int(r.get("duration_ms", 0)),
            "Time": r.get("ts", "")[:19].replace("T", " "),
        })
    st.dataframe(rows, width="stretch")

    with st.expander("Candidate scores per document (all templates vs each file)"):
        for r in records:
            cands = r.get("candidate_scores", {})
            if not cands:
                continue
            st.markdown(f"**{r.get('file')}** — best: `{r.get('template_id') or 'MISS'}` ({r.get('match_score', 0):.3f})")
            score_rows = [{"Template": tid, "Score": f"{s:.4f}"} for tid, s in sorted(cands.items(), key=lambda x: -x[1])]
            st.dataframe(score_rows, width="stretch", hide_index=True)

    raw_lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    st.download_button(
        "Download audit log (JSONL)",
        data=raw_lines,
        file_name="extraction_audit.jsonl",
        mime="application/jsonlines",
    )


def _render_template_editor() -> None:
    """Template store CRUD editor."""
    from doc_schema_extractor.template_store import TemplateStore
    from doc_schema_extractor.models import Template

    store = TemplateStore()
    templates = store.list_all()

    st.subheader("Stored templates")
    if not templates:
        st.info("No templates in store yet.")
    else:
        template_ids = [t.template_id for t in templates]
        selected_id = st.selectbox("Select template to view / edit / delete", template_ids)
        tmpl = store.get(selected_id)
        if tmpl:
            col_view, col_del = st.columns([4, 1])
            with col_view:
                edited_raw = st.text_area(
                    "Template JSON (edit and press Save to update)",
                    value=tmpl.model_dump_json(indent=2),
                    height=420,
                    key=f"editor_{selected_id}",
                )
            with col_del:
                st.write("")
                st.write("")
                if st.button("🗑️ Delete", key=f"del_{selected_id}"):
                    store.delete(selected_id)
                    st.success(f"Deleted `{selected_id}`")
                    st.rerun()

            if st.button("💾 Save changes", key=f"save_{selected_id}"):
                try:
                    parsed = json.loads(edited_raw)
                    updated = Template.model_validate(parsed)
                    store.add(updated)
                    st.success(f"Saved `{updated.template_id}`")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Invalid JSON or schema error: {exc}")

    st.divider()
    st.subheader("Add / import template")
    st.caption("Paste a full template JSON object (single template, not the whole store dict).")
    new_raw = st.text_area("Paste template JSON here", height=300, key="new_template_input")
    if st.button("➕ Import template"):
        if not new_raw.strip():
            st.warning("Nothing to import.")
        else:
            try:
                parsed = json.loads(new_raw)
                new_tmpl = Template.model_validate(parsed)
                store.add(new_tmpl)
                st.success(f"Imported `{new_tmpl.template_id}` — {len(new_tmpl.extraction_rules)} rules, {len(new_tmpl.fingerprint.required_keywords)} keywords")
                st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")

    st.divider()
    st.subheader("Download full store")
    all_data = {t.template_id: json.loads(t.model_dump_json()) for t in store.list_all()}
    st.download_button(
        "⬇️ Download store.json",
        data=json.dumps(all_data, indent=2, ensure_ascii=False),
        file_name="store.json",
        mime="application/json",
    )


def main():
    st.set_page_config(page_title="Doc Schema Extractor", page_icon="📄", layout="wide")
    st.title("📄 Doc Schema Extractor")
    logger.info("Streamlit UI started")

    with st.sidebar:
        st.header("⚙️ Config")
        default_backend = _default_backend()
        backend_index = 0 if default_backend == "ollama" else 1
        backend_name = st.selectbox("LLM Backend", ["ollama", "openai"], index=backend_index)
        model = st.selectbox(
            "Model",
            OLLAMA_MODELS if backend_name == "ollama" else OPENAI_MODELS,
        )
        threshold = st.slider("Match threshold", 0.50, 0.95, 0.75, 0.05)
        if backend_name == "ollama":
            timeout_val = st.number_input(
                "Ollama timeout (s)", min_value=30, max_value=900,
                value=int(os.getenv("OLLAMA_TIMEOUT", "300")), step=30,
            )
        else:
            timeout_val = 300
        st.divider()
        langsmith_on = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        if langsmith_on:
            st.success(f"LangSmith ON\n`{os.getenv('LANGSMITH_PROJECT', '—')}`")
        else:
            st.warning("LangSmith OFF")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "extraction_result" not in st.session_state:
        st.session_state.extraction_result = None

    tab_extract, tab_scores, tab_templates = st.tabs(["Extract", "Score History", "📁 Templates"])

    with tab_extract:
        uploaded = st.file_uploader("Upload PDF or XLSX", type=["pdf", "xlsx"])
        if uploaded:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            logger.info("Uploaded file name=%s temp_path=%s", uploaded.name, tmp_path)

            if st.button("▶ Run Extraction", type="primary"):
                from doc_schema_extractor import Extractor
                from doc_schema_extractor.backends import OllamaBackend, OpenAIBackend

                be = (
                    OpenAIBackend(model=model)
                    if backend_name == "openai"
                    else OllamaBackend(model=model, timeout=float(timeout_val))
                )
                extractor = Extractor(backend=be, match_threshold=threshold)
                with st.spinner("Extracting..."):
                    result = extractor.extract(tmp_path)

                st.session_state.extraction_result = json.loads(result.model_dump_json())
                st.session_state.messages = []
                badge = "🟡 LLM" if result.llm_used else "🟢 Template HIT"
                logger.info(
                    "UI extraction complete template_id=%s llm_used=%s score=%.3f",
                    result.template_id, result.llm_used, result.match_score,
                )
                st.success(
                    f"{badge} | template=`{result.template_id}` | "
                    f"score={result.match_score:.2f} | "
                    f"valid={'✓' if result.validation_passed else '✗'}"
                )

        if st.session_state.extraction_result:
            er = st.session_state.extraction_result
            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader("Extracted data")
                st.json(er.get("data", {}))
            with col2:
                st.subheader("Run metadata")
                st.json({
                    "template_id": er.get("template_id"),
                    "match_score": er.get("match_score"),
                    "llm_used": er.get("llm_used"),
                    "llm_backend": er.get("llm_backend"),
                    "llm_model": er.get("llm_model"),
                    "validation_passed": er.get("validation_passed"),
                    "validation_errors": er.get("validation_errors"),
                })

            st.divider()
            st.subheader("💬 Chat with this document")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if user_input := st.chat_input("Ask anything about this document..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = answer_question(user_input, er, backend_name, model)
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

    with tab_scores:
        if st.button("🔄 Refresh"):
            st.rerun()
        _render_score_dashboard()

    with tab_templates:
        if st.button("🔄 Refresh", key="tmpl_refresh"):
            st.rerun()
        _render_template_editor()


if __name__ == "__main__":
    main()
