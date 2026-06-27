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
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "o4-mini"]


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
    trace_chat_turn(
        question=question,
        template_id=extraction_result.get("template_id"),
        backend=backend_name, model=model,
    )
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
        return _answer_openai(prompt, model)
    return _answer_ollama(prompt, model)


def _render_score_dashboard() -> None:
    """Show cross-document score table from the audit log."""
    from doc_schema_extractor.audit_log import AuditLog
    audit = AuditLog()
    records = audit.read_all()
    if not records:
        st.info("No extraction runs recorded yet.")
        return

    st.subheader("Cross-document extraction scores")

    # Summary table
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

    st.dataframe(rows, use_container_width=True)

    # Per-document candidate score breakdown
    with st.expander("Candidate scores per document (all templates vs each file)"):
        for r in records:
            cands = r.get("candidate_scores", {})
            if not cands:
                continue
            st.markdown(f"**{r.get('file')}** — best: `{r.get('template_id') or 'MISS'}` ({r.get('match_score', 0):.3f})")
            score_rows = [{"Template": tid, "Score": f"{s:.4f}"} for tid, s in sorted(cands.items(), key=lambda x: -x[1])]
            st.dataframe(score_rows, use_container_width=True, hide_index=True)

    # Raw JSONL download
    raw_lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    st.download_button(
        "Download audit log (JSONL)",
        data=raw_lines,
        file_name="extraction_audit.jsonl",
        mime="application/jsonlines",
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

    tab_extract, tab_scores = st.tabs(["Extract", "Score History"])

    # ── Extract tab ───────────────────────────────────────────────────────
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

    # ── Score history tab ─────────────────────────────────────────────────
    with tab_scores:
        if st.button("🔄 Refresh"):
            st.rerun()
        _render_score_dashboard()


if __name__ == "__main__":
    main()
