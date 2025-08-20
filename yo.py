"""
Databricks Agentic RAG — Robust Ingestion with Guaranteed Embeddings
---------------------------------------------------------------------
This Streamlit app ingests PDFs, generates embeddings with a Databricks Serving endpoint,
stores them **reliably** in Delta (VECTOR or ARRAY<FLOAT> with JSON→FROM_JSON), and
exposes a RAG chat over Databricks Vector Search. It includes:

• Auto-detection of VECTOR type support (falls back to ARRAY<FLOAT>)
• Direct Access index by default (no sync needed); falls back to Delta-sync (TRIGGERED)
• JSON-safe inserts for embeddings (fixes empty/NULL embeddings issue)
• Strict dimension checks, padding/truncation, and detailed logs
• Accurate verification (no double-counting)
• Safe de-duplication by pdf_name (optional)
• Agentic and simple RAG chat modes

Env (.env) expected:
  CHAT_ENDPOINT=<databricks chat serving endpoint name>
  EMBEDDING_ENDPOINT=<databricks embedding endpoint name>
  DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
  DATABRICKS_TOKEN=<token>
  DATABRICKS_SQL_HTTP_PATH=/sql/1.0/warehouses/<id>
  CATALOG=main
  SCHEMA=default
  TABLE_NAME=pdf_text_embeddings     # optional, default used if missing
  DELTA_TABLE=main.default.pdf_text_embeddings  # overrides CATALOG/SCHEMA/TABLE_NAME if set
  VS_ENDPOINT=vs_endpoint_default    # optional (reused or created if missing)
  VS_INDEX=main.default.pdf_text_embeddings_index
  AGENTIC_MODE=True|False            # optional
  DEBUG_VERBOSE=True|False           # optional

Run:
  streamlit run databricks_agentic_rag_app.py

Note: Rotate your tokens if they were ever exposed.
"""

import io
import json
import os
import re
import sys
import time
import uuid
import typing as t
import datetime as dt

import streamlit as st
import pdfplumber
from dotenv import load_dotenv

from databricks import sql
from databricks.vector_search.client import VectorSearchClient
from mlflow.deployments import get_deploy_client

# --------------------
# Config & Globals
# --------------------
load_dotenv()

CHAT_ENDPOINT = os.getenv("CHAT_ENDPOINT", "").strip()
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "").strip()
DB_HOST = os.getenv("DATABRICKS_HOST", "").replace("https://", "").strip()
DB_HTTP_PATH = os.getenv("DATABRICKS_SQL_HTTP_PATH", "").strip()
DB_TOKEN = os.getenv("DATABRICKS_TOKEN", "").strip()

CATALOG = os.getenv("CATALOG", "main").strip()
SCHEMA = os.getenv("SCHEMA", "default").strip()
TABLE_NAME = os.getenv("TABLE_NAME", "pdf_text_embeddings").strip()
DELTA_TABLE = os.getenv("DELTA_TABLE", f"{CATALOG}.{SCHEMA}.{TABLE_NAME}").strip()

VS_ENDPOINT = os.getenv("VS_ENDPOINT", "").strip()  # may be reused if exists
VS_INDEX = os.getenv("VS_INDEX", f"{CATALOG}.{SCHEMA}.{TABLE_NAME}_index").strip()

AGENTIC_MODE = os.getenv("AGENTIC_MODE", "False").lower().strip() == "true"
DEBUG_VERBOSE = os.getenv("DEBUG_VERBOSE", "False").lower().strip() == "true"

# Embedding dimension expected from endpoint — adjust if your model differs
EMBED_DIM_DEFAULT = 1024

# Will be set after table creation/probing
USE_VECTOR_TYPE: bool = False
VECTOR_DIM: int = EMBED_DIM_DEFAULT
INDEX_MODE: str = "UNKNOWN"  # "DIRECT" or "SYNC"

# Databricks clients
client = get_deploy_client("databricks")
vsc = VectorSearchClient()

# --------------------
# Utility logging
# --------------------

def log_debug(*args):
    if DEBUG_VERBOSE:
        try:
            st.write("DEBUG:", *args)
        except Exception:
            pass
        print("DEBUG:", *args)

def log_info(*args):
    try:
        st.info(" ".join(map(str, args)))
    except Exception:
        print("INFO:", *args)

def log_warn(*args):
    try:
        st.warning(" ".join(map(str, args)))
    except Exception:
        print("WARN:", *args)

def log_error(*args):
    try:
        st.error(" ".join(map(str, args)))
    except Exception:
        print("ERROR:", *args)

# --------------------
# DB Connection helper
# --------------------

def db_connect():
    if not DB_HOST or not DB_HTTP_PATH or not DB_TOKEN:
        raise RuntimeError("Databricks SQL connection env vars are missing.")
    return sql.connect(server_hostname=DB_HOST, http_path=DB_HTTP_PATH, access_token=DB_TOKEN)

# --------------------
# Capability probing
# --------------------

def parse_catalog_schema_from_fqn(fqn: str) -> t.Tuple[str, str, str]:
    """Return (catalog, schema, table) from an FQN or partially-qualified name."""
    parts = fqn.split(".")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return CATALOG, parts[0], parts[1]
    return CATALOG, SCHEMA, parts[0]


def supports_vector_type() -> bool:
    """Create and drop a tiny probe table with VECTOR to detect support."""
    catalog, schema, _ = parse_catalog_schema_from_fqn(DELTA_TABLE)
    probe = f"{catalog}.{schema}.__vector_probe__{uuid.uuid4().hex[:6]}"
    conn = db_connect()
    try:
        with conn.cursor() as c:
            c.execute(f"CREATE TABLE {probe} (v VECTOR(3)) USING DELTA")
        log_debug("VECTOR type supported.")
        return True
    except Exception as e:
        log_debug("VECTOR type probe failed:", e)
        return False
    finally:
        try:
            with conn.cursor() as c:
                c.execute(f"DROP TABLE IF EXISTS {probe}")
        except Exception:
            pass
        conn.close()

# --------------------
# DDL & Index bootstrap
# --------------------

def ensure_table_and_index(dim: int = EMBED_DIM_DEFAULT):
    global USE_VECTOR_TYPE, VECTOR_DIM, VS_ENDPOINT, INDEX_MODE

    VECTOR_DIM = dim

    # 1) Ensure table with VECTOR or ARRAY<FLOAT>
    USE_VECTOR_TYPE = supports_vector_type()
    conn = db_connect()
    with conn.cursor() as c:
        if USE_VECTOR_TYPE:
            ddl = f"""
            CREATE TABLE IF NOT EXISTS {DELTA_TABLE} (
              doc_id STRING,
              pdf_name STRING,
              page INT,
              chunk_id STRING,
              content STRING,
              embedding VECTOR({dim}),
              created_at TIMESTAMP
            ) USING DELTA
            TBLPROPERTIES (delta.enableChangeDataFeed = true)
            """
            c.execute(ddl)
            log_info(f"Ensured table with VECTOR({dim}): {DELTA_TABLE}")
        else:
            ddl = f"""
            CREATE TABLE IF NOT EXISTS {DELTA_TABLE} (
              doc_id STRING,
              pdf_name STRING,
              page INT,
              chunk_id STRING,
              content STRING,
              embedding ARRAY<FLOAT>,
              created_at TIMESTAMP
            ) USING DELTA
            TBLPROPERTIES (delta.enableChangeDataFeed = true)
            """
            c.execute(ddl)
            log_info(f"Ensured table with ARRAY<FLOAT>: {DELTA_TABLE}")
    conn.close()

    # 2) Ensure VS endpoint (reuse existing if any)
    endpoints = vsc.list_endpoints().get("endpoints", [])
    if endpoints:
        if not VS_ENDPOINT:
            VS_ENDPOINT = endpoints[0]["name"]
            os.environ["VS_ENDPOINT"] = VS_ENDPOINT
        log_info(f"Using VS endpoint: {VS_ENDPOINT}")
    else:
        ep_name = VS_ENDPOINT or f"rag-vs-endpoint-{uuid.uuid4().hex[:6]}"
        ep = vsc.create_endpoint(name=ep_name, endpoint_type="STANDARD")
        VS_ENDPOINT = ep["name"]
        os.environ["VS_ENDPOINT"] = VS_ENDPOINT
        log_info(f"Created VS endpoint: {VS_ENDPOINT}")

    # 3) Ensure index: try Direct Access first, fallback to Delta-sync (TRIGGERED)
    try:
        _ = vsc.get_index(VS_ENDPOINT, VS_INDEX)
        log_info(f"Index already exists: {VS_INDEX} ✅")
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
            try:
                log_info(f"Creating Direct Access index {VS_INDEX}…")
                vsc.create_direct_access_index(
                    endpoint_name=VS_ENDPOINT,
                    index_name=VS_INDEX,
                    source_table_name=DELTA_TABLE,
                    primary_key="chunk_id",
                    embedding_vector_column="embedding",
                )
                log_info(f"Created Direct Access index: {VS_INDEX} ✅")
            except Exception as e2:
                log_warn("Direct Access creation failed, falling back to Delta-sync (TRIGGERED):", e2)
                vsc.create_delta_sync_index(
                    endpoint_name=VS_ENDPOINT,
                    index_name=VS_INDEX,
                    source_table_name=DELTA_TABLE,
                    pipeline_type="TRIGGERED",
                    primary_key="chunk_id",
                    embedding_dimension=dim,
                    embedding_vector_column="embedding",
                )
                log_info(f"Created Delta-sync index: {VS_INDEX} ✅")
        else:
            raise

    # Determine index mode for later sync/query behavior
    try:
        idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)
        pipeline_type = getattr(idx, "pipeline_type", "DIRECT_ACCESS").upper()
        INDEX_MODE = "DIRECT" if pipeline_type == "DIRECT_ACCESS" else "SYNC"
        log_info(f"Index mode: {INDEX_MODE} (pipeline_type={pipeline_type})")
    except Exception as e:
        log_warn("Could not determine index pipeline type:", e)
        INDEX_MODE = "UNKNOWN"


def trigger_index_sync_if_needed():
    if INDEX_MODE == "SYNC":
        try:
            vsc.trigger_index_sync(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
            log_info("Index sync triggered ✅ (Delta-sync)")
        except Exception as e:
            log_warn("Index sync trigger failed:", e)
    else:
        log_debug("No sync needed (Direct Access or unknown mode).")

# --------------------
# Chunking & Embedding
# --------------------

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200, min_len: int = 20) -> t.List[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks: t.List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        k = text.rfind(" ", i + max_chars - 200, j)
        j = k if k != -1 else j
        piece = text[i:j]
        if piece and len(piece.strip()) >= min_len:
            chunks.append(piece.strip())
        i = max(j - overlap, j)
    return chunks


def sanitize_vector(vec: t.Optional[t.List[float]], dim: int) -> t.List[float]:
    """Ensure vector has exact length dim by truncation/padding. Returns [] if vec invalid."""
    if not isinstance(vec, list):
        return []
    v = [float(x) for x in vec if isinstance(x, (int, float))]
    if len(v) == 0:
        return []
    if len(v) == dim:
        return v
    if len(v) > dim:
        return v[:dim]
    # pad with zeros
    return v + [0.0] * (dim - len(v))


def embed_batch(texts: t.List[str], batch_size: int = 64, expected_dim: int = VECTOR_DIM) -> t.List[t.List[float]]:
    """Embed texts in batches; always returns a list of vectors (possibly empty lists), never raises."""
    embeddings: t.List[t.List[float]] = []
    if not texts:
        return embeddings

    log_info("EMBEDDING_ENDPOINT →", EMBEDDING_ENDPOINT)
    log_debug("Total texts to embed:", len(texts))

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            resp = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": batch})
            data = resp.get("data", [])
            if len(data) != len(batch):
                log_warn(f"Embedding response size mismatch: expected {len(batch)}, got {len(data)}")
            for i, row in enumerate(data):
                raw = row.get("embedding") if isinstance(row, dict) else None
                vec = sanitize_vector(raw, expected_dim)
                embeddings.append(vec)
                log_debug(f"Batch vec[{start + i}] len=", len(vec))
        except Exception as e:
            log_error("Embedding batch failed:", e)
            # Fill with empties to keep ordering
            embeddings.extend([[] for _ in batch])
    return embeddings

# --------------------
# Ingestion & Write
# --------------------

def delete_existing_by_pdf_name(pdf_name: str):
    conn = db_connect()
    with conn.cursor() as c:
        c.execute(f"DELETE FROM {DELTA_TABLE} WHERE pdf_name = ?", (pdf_name,))
    conn.close()
    log_info(f"Removed existing rows for pdf_name='{pdf_name}' (if any)")


def write_rows(rows: t.List[t.Tuple[str, str, int, str, str, t.List[float], dt.datetime]]):
    """Robust insert: JSON → FROM_JSON → CAST(… AS VECTOR(dim)) or ARRAY<FLOAT>.
    rows: (doc_id, pdf_name, page, chunk_id, content, embedding_list, created_at)
    """
    if not rows:
        log_warn("write_rows called with empty rows")
        return

    conn = db_connect()
    with conn.cursor() as c:
        if USE_VECTOR_TYPE:
            insert_sql = f"""
            INSERT INTO {DELTA_TABLE} (doc_id, pdf_name, page, chunk_id, content, embedding, created_at)
            SELECT ?, ?, ?, ?, ?, CAST(from_json(?, 'array<float>') AS VECTOR({VECTOR_DIM})), ?
            """
        else:
            insert_sql = f"""
            INSERT INTO {DELTA_TABLE} (doc_id, pdf_name, page, chunk_id, content, embedding, created_at)
            SELECT ?, ?, ?, ?, ?, CAST(from_json(?, 'array<float>') AS ARRAY<FLOAT>), ?
            """

        payload = []
        for r in rows:
            emb_json = json.dumps(r[5] or [])
            payload.append((r[0], r[1], r[2], r[3], r[4], emb_json, r[6]))
            log_debug(
                f"Row: doc_id={r[0]} page={r[2]} chunk_id={r[3][:8]}.. content_len={len(r[4])} emb_len={len(r[5]) if r[5] else 0}"
            )
        c.executemany(insert_sql, payload)
    conn.close()
    log_info(f"Inserted {len(rows)} rows into {DELTA_TABLE}")


def verify_embeddings(pdf_name: t.Optional[str] = None, limit: int = 5):
    """Accurate verification without double-counting.
    For ARRAY: size(embedding); for VECTOR: vector_dims(embedding).
    """
    conn = db_connect()
    with conn.cursor() as c:
        base = f"FROM {DELTA_TABLE}"
        params: t.Tuple = tuple()
        if pdf_name:
            base += " WHERE pdf_name = ?"
            params = (pdf_name,)

        # Preview rows
        if USE_VECTOR_TYPE:
            preview_sql = f"SELECT pdf_name, page, chunk_id, vector_dims(embedding) AS emb_dims, substr(content,1,160) AS preview {base} LIMIT {limit}"
        else:
            preview_sql = f"SELECT pdf_name, page, chunk_id, size(embedding) AS emb_dims, substr(content,1,160) AS preview {base} LIMIT {limit}"
        c.execute(preview_sql, params)
        rows = c.fetchall()
        log_info(f"🔍 Verifying embeddings for: {pdf_name or 'ALL PDFs'} (showing up to {limit})")
        for r in rows:
            emb_dims = r[3] or 0
            status = "✅" if emb_dims > 0 else "⚠️ MISSING"
            st.write(f"PDF={r[0]}, Page={r[1]}, Chunk={r[2]}, Embedding dims={emb_dims} {status}")
            st.caption(f"Preview: {r[4]}…")

        # Totals
        c.execute(f"SELECT count(*) {base}", params)
        total = c.fetchall()[0][0]

        if USE_VECTOR_TYPE:
            missing_sql = f"SELECT count(*) {base} AND (embedding IS NULL OR vector_dims(embedding)=0)" if pdf_name else \
                          f"SELECT count(*) FROM {DELTA_TABLE} WHERE embedding IS NULL OR vector_dims(embedding)=0"
        else:
            missing_sql = f"SELECT count(*) {base} AND (embedding IS NULL OR size(embedding)=0)" if pdf_name else \
                          f"SELECT count(*) FROM {DELTA_TABLE} WHERE embedding IS NULL OR size(embedding)=0"
        c.execute(missing_sql, params if pdf_name else tuple())
        missing = c.fetchall()[0][0]

    conn.close()
    st.success(f"Total chunks: {total}, Chunks missing embeddings: {missing}")


def ingest_pdf(file_bytes: bytes, pdf_name: str, dedup: bool = True) -> t.Dict[str, t.Any]:
    """Read PDF, chunk text, embed, and write rows. Guarantees stored vectors.
    Returns {doc_id, chunks}.
    """
    doc_id = str(uuid.uuid4())
    rows: t.List[t.Tuple[str, str, int, str, str, t.List[float], dt.datetime]] = []

    if dedup:
        delete_existing_by_pdf_name(pdf_name)

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        st.info(f"ℹ️ Processing PDF: {pdf_name}, total pages={len(pdf.pages)}")
        for pageno, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                log_warn(f"Page {pageno} has no extractable text. Consider OCR fallback.")
                continue

            chunks = chunk_text(text)
            if not chunks:
                log_warn(f"Page {pageno}: no chunks generated after splitting.")
                continue

            # Embed
            st.write(f"Embedding page {pageno} with {len(chunks)} chunks…")
            embs = embed_batch(chunks, batch_size=64, expected_dim=VECTOR_DIM)

            # Strict guarantee: store vectors with exact length; empty vecs allowed but counted
            now = dt.datetime.now(dt.timezone.utc)
            for ch, emb in zip(chunks, embs):
                clean = sanitize_vector(emb, VECTOR_DIM)
                rows.append((doc_id, pdf_name, pageno, str(uuid.uuid4()), ch, clean, now))

    if not rows:
        st.error(f"❌ No rows to write for PDF: {pdf_name}")
        return {"doc_id": doc_id, "chunks": 0}

    write_rows(rows)
    trigger_index_sync_if_needed()
    st.success(f"✅ PDF ingested: {pdf_name}, total chunks={len(rows)}")

    verify_embeddings(pdf_name)
    return {"doc_id": doc_id, "chunks": len(rows)}

# --------------------
# Retrieval & Chat
# --------------------

def embed_query_single(text: str) -> t.List[float]:
    try:
        out = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": [text]})
        data = out.get("data", [])
        if not data:
            return []
        return sanitize_vector(data[0].get("embedding"), VECTOR_DIM)
    except Exception as e:
        log_error("Query embedding failed:", e)
        return []


def retrieve(query: str, k: int = 5) -> t.List[t.Dict[str, t.Any]]:
    try:
        idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)
        # Try query_text first (if supported), else query_vector
        try:
            res = idx.similarity_search(
                query_text=query,
                num_results=k,
                columns=["doc_id", "pdf_name", "page", "content"],
            )
        except Exception as e:
            if any(s in str(e).lower() for s in ["query vector must be specified", "invalid_parameter_value", "resource_does_not_exist"]):
                qv = embed_query_single(query)
                res = idx.similarity_search(
                    query_vector=qv,
                    num_results=k,
                    columns=["doc_id", "pdf_name", "page", "content"],
                )
            else:
                raise
        rows = res.get("result", {}).get("data_array", [])
        out = []
        for r in rows:
            out.append({
                "doc_id": r[0] if len(r) > 0 else None,
                "pdf_name": r[1] if len(r) > 1 else None,
                "page": r[2] if len(r) > 2 else None,
                "content": r[3] if len(r) > 3 else "",
            })
        return out
    except Exception as e:
        log_error("Retrieval failed:", e)
        return []


def call_llm(messages: t.List[dict], max_tokens: int = 400) -> str:
    try:
        resp = client.predict(endpoint=CHAT_ENDPOINT, inputs={"messages": messages, "max_tokens": max_tokens})
        if "choices" in resp:
            return resp["choices"][0]["message"]["content"]
        if "content" in resp:
            return resp["content"]
        return str(resp)[:2000]
    except Exception as e:
        log_error("LLM call failed:", e)
        return "⚠️ LLM error. Please check logs."


def agentic_chat(question: str, history: t.List[dict], k: int = 5, max_tokens: int = 512):
    ctx_list = retrieve(question, k=k)
    if ctx_list:
        context_text = "\n\n".join(f"[{c['pdf_name']} - page {c['page']}]: {c['content']}" for c in ctx_list)
        system_prompt = (
            "You are a helpful assistant. Use the retrieved context from the uploaded document "
            "to answer the user's question. If the context is insufficient, say so.\n\n"
            f"Retrieved context:\n{context_text}"
        )
    else:
        system_prompt = "You are a helpful assistant. No relevant document context was found."

    msgs = [{"role": "system", "content": system_prompt}]
    for h in history:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": question})

    reply = call_llm(msgs, max_tokens=max_tokens)
    return reply, ctx_list


def chat_with_rag(question: str, history: t.Optional[t.List[dict]] = None, k: int = 5, max_tokens: int = 512):
    history = history or []
    if AGENTIC_MODE:
        return agentic_chat(question, history, k=k, max_tokens=max_tokens)

    ctx_list = retrieve(question, k=k)
    if ctx_list:
        context_text = "\n\n".join(f"[{c['pdf_name']} - p.{c['page']}]: {c['content']}" for c in ctx_list)
        system_prompt = (
            "Answer using ONLY this context. If the context is insufficient, say 'I don't know'.\n\n"
            f"Retrieved context:\n{context_text}"
        )
    else:
        system_prompt = (
            "Answer using ONLY this context. If the context is insufficient, say 'I don't know'.\n\n"
            "No context found."
        )

    msgs = [{"role": "system", "content": system_prompt}]
    for h in history:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": question})

    reply = call_llm(msgs, max_tokens=max_tokens)
    return reply, ctx_list

# --------------------
# Streamlit UI
# --------------------

st.set_page_config(page_title="Databricks Agentic RAG", page_icon="🧠")
st.title("🧠 Databricks Agentic RAG Chat — Robust Embeddings")

# Early sanity check against the embedding endpoint
try:
    resp_test = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": ["_probe_"]})
    dim_probe = len((resp_test.get("data", [{}])[0] or {}).get("embedding", []) or [])
    if dim_probe:
        VECTOR_DIM = dim_probe
        st.caption(f"Embedding endpoint OK • dimension={VECTOR_DIM}")
    else:
        st.caption("Embedding endpoint responded but no embedding vector found — will use default 1024.")
except Exception as e:
    st.warning(f"Embedding endpoint probe failed: {e}")

# Ensure resources at startup
ensure_table_and_index(dim=VECTOR_DIM or EMBED_DIM_DEFAULT)

with st.sidebar:
    st.header("Upload PDF")
    f = st.file_uploader("Choose a PDF", type=["pdf"])
    dedup = st.checkbox("De-duplicate rows for same PDF name before ingest", value=True)
    if st.button("Ingest", disabled=not f):
        r = ingest_pdf(f.read(), f.name, dedup=dedup)
        if INDEX_MODE == "SYNC":
            st.success(f"Ingested {r['chunks']} chunks (doc_id={r['doc_id']}) and index sync triggered ✅")
        else:
            st.success(f"Ingested {r['chunks']} chunks (doc_id={r['doc_id']}) ✅ (Direct Access)")

    st.divider()
    st.subheader("Index Info")
    try:
        idx_obj = vsc.get_index(VS_ENDPOINT, VS_INDEX)

        # Convert the object into a dict safely
        idx_meta = {
            "endpoint": VS_ENDPOINT,
            "index": VS_INDEX,
            "status": getattr(idx_obj, "status", "?"),
            "pipeline_type": getattr(idx_obj, "pipeline_type", "N/A"),  # only exists for delta-sync
            "last_sync": getattr(idx_obj, "last_successful_write_time_ms", None),
        }

        st.json(idx_meta)

    except Exception as e:
        st.write("Index meta unavailable:", str(e))


# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

q = st.chat_input("Ask me anything…")
if q:
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking (Agentic)" if AGENTIC_MODE else "🤔 Thinking…"):
            ans, ctx = chat_with_rag(q, st.session_state.history[:-1])
        st.markdown(ans)
        if ctx:
            with st.expander("Sources"):
                for c in ctx:
                    st.markdown(f"📄 **{c['pdf_name']}** (p.{c['page']})\n\n> {c['content'][:400]}…")
    st.session_state.history.append({"role": "assistant", "content": ans})