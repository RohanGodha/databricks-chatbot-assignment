"""
Databricks Agentic RAG — Robust Ingestion with Guaranteed Embeddings
---------------------------------------------------------------------
Single-file Streamlit app that:
• Uploads PDFs
• Extracts/chunks text
• Creates embeddings (Databricks OR OpenAI)
• Stores safely into Delta (VECTOR or ARRAY<FLOAT> via JSON→FROM_JSON)
• Builds/uses Databricks Vector Search (Direct Access preferred)
• Runs a RAG chat

Fixes:
• Empty/NULL embeddings via strict sanitization + JSON casting
• Robust index metadata (.getattr)
• Auto-detects VECTOR type, falls back to ARRAY<FLOAT>
• Accurate verification

ENV (.env):
  # Choose backend: DATABRICKS or OPENAI
  BACKEND_PROVIDER=DATABRICKS

  # Databricks config
  DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
  DATABRICKS_TOKEN=<token>
  DATABRICKS_SQL_HTTP_PATH=/sql/1.0/warehouses/<id>

  # Databricks Serving endpoints (used when BACKEND_PROVIDER=DATABRICKS)
  CHAT_ENDPOINT=<databricks chat serving endpoint name>
  EMBEDDING_ENDPOINT=<databricks embedding endpoint name>

  # OpenAI config (used when BACKEND_PROVIDER=OPENAI)
  OPENAI_API_KEY=<key>
  OPENAI_EMBED_MODEL=text-embedding-3-small
  OPENAI_CHAT_MODEL=gpt-4o-mini

  # Data placement
  CATALOG=main
  SCHEMA=default
  TABLE_NAME=pdf_text_embeddings
  DELTA_TABLE=main.default.pdf_text_embeddings
  VS_ENDPOINT=vs_endpoint_default
  VS_INDEX=main.default.pdf_text_embeddings_index

  # Optional
  AGENTIC_MODE=True
  DEBUG_VERBOSE=True
"""

import io
import json
import os
import uuid
import typing as t
import datetime as dt
import re


import streamlit as st
import pdfplumber
from dotenv import load_dotenv

from databricks import sql
from databricks.vector_search.client import VectorSearchClient
from mlflow.deployments import get_deploy_client
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer



# --------------------
# Config & Globals
# --------------------
load_dotenv()

BACKEND_PROVIDER = os.getenv("BACKEND_PROVIDER", "DATABRICKS").strip().upper()

# Databricks
DB_HOST = os.getenv("DATABRICKS_HOST", "").replace("https://", "").strip()
DB_HTTP_PATH = os.getenv("DATABRICKS_SQL_HTTP_PATH", "").strip()
DB_TOKEN = os.getenv("DATABRICKS_TOKEN", "").strip()
CHAT_ENDPOINT = os.getenv("CHAT_ENDPOINT", "").strip()
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "").strip()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()

# Data placement
CATALOG = os.getenv("CATALOG", "main").strip()
SCHEMA = os.getenv("SCHEMA", "default").strip()
TABLE_NAME = os.getenv("TABLE_NAME", "pdf_text_embeddings").strip()
DELTA_TABLE = os.getenv("DELTA_TABLE", f"{CATALOG}.{SCHEMA}.{TABLE_NAME}").strip()

VS_ENDPOINT = os.getenv("VS_ENDPOINT", "").strip()
VS_INDEX = os.getenv("VS_INDEX", f"{CATALOG}.{SCHEMA}.{TABLE_NAME}_index").strip()

AGENTIC_MODE = os.getenv("AGENTIC_MODE", "False").lower().strip() == "true"
DEBUG_VERBOSE = os.getenv("DEBUG_VERBOSE", "False").lower().strip() == "true"

# Default dims (will probe)
EMBED_DIM_DEFAULT = 1024

# Will be set after probe
USE_VECTOR_TYPE: bool = False
VECTOR_DIM: int = EMBED_DIM_DEFAULT
INDEX_MODE: str = "UNKNOWN"  # "DIRECT" or "SYNC"

# Global variables (adjust as per your setup)
if BACKEND_PROVIDER == "LOCAL":
    EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    DIM = EMBED_MODEL.get_sentence_embedding_dimension()
else:
    EMBED_MODEL = None
    DIM = VECTOR_DIM
INDEX = faiss.IndexFlatIP(DIM)  
DOC_STORE = []

# At the top of your script, after the global declarations:
if "doc_store" not in st.session_state:
    st.session_state.doc_store = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatIP(DIM)

# Then update all references:
DOC_STORE = st.session_state.doc_store
INDEX = st.session_state.faiss_index

# Clients
vsc = VectorSearchClient()

# Backends
dbx_client = None
oa_client = None
if BACKEND_PROVIDER == "DATABRICKS":
    dbx_client = get_deploy_client("databricks")
elif BACKEND_PROVIDER == "OPENAI":
    try:
        from openai import OpenAI
        oa_client = OpenAI(api_key=OPENAI_API_KEY or None)
    except Exception as e:
        raise RuntimeError(f"OPENAI selected but OpenAI SDK not available: {e}")
else:
    raise RuntimeError("BACKEND_PROVIDER must be DATABRICKS or OPENAI")

# --------------------
# Logging helpers
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
# DB connection
# --------------------
def db_connect():
    if not DB_HOST or not DB_HTTP_PATH or not DB_TOKEN:
        raise RuntimeError("Databricks SQL connection env vars are missing.")
    return sql.connect(server_hostname=DB_HOST, http_path=DB_HTTP_PATH, access_token=DB_TOKEN)

# --------------------
# Capability probing
# --------------------
def parse_catalog_schema_from_fqn(fqn: str) -> t.Tuple[str, str, str]:
    parts = fqn.split(".")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return CATALOG, parts[0], parts[1]
    return CATALOG, SCHEMA, parts[0]

def supports_vector_type() -> bool:
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

    # 1) Table
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

    # 2) VS endpoint
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

    # 3) Index (Direct Access → Delta-sync fallback)
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

    # Index mode
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
# Chunking
# --------------------
# def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200, min_len: int = 20) -> t.List[str]:
#     text = " ".join((text or "").split())
#     if not text:
#         return []
#     chunks: t.List[str] = []
#     i = 0
#     n = len(text)
#     while i < n:
#         j = min(i + max_chars, n)
#         k = text.rfind(" ", i + max_chars - 200, j)
#         j = k if k != -1 else j
#         piece = text[i:j]
#         if piece and len(piece.strip()) >= min_len:
#             chunks.append(piece.strip())
#         i = max(j - overlap, j)
#     return chunks


# def chunk_text(text: str, max_chars: int = 800, overlap: int = 150, min_len: int = 50) -> t.List[str]:
#     """Better chunking with sentence awareness"""
#     if not text or not text.strip():
#         return []
    
#     # Clean the text first
#     text = re.sub(r'\s+', ' ', text.strip())
    
#     chunks = []
#     start = 0
#     n = len(text)
    
#     while start < n:
#         # Try to find a good break point (sentence end, paragraph, etc.)
#         end = start + max_chars
        
#         # Don't break in the middle of a word if possible
#         if end < n:
#             # Look for sentence endings first
#             for break_char in ['. ', '! ', '? ', '\n', '; ']:
#                 break_pos = text.rfind(break_char, start + min_len, end)
#                 if break_pos != -1:
#                     end = break_pos + len(break_char.strip())
#                     break
#             else:
#                 # Fallback: break at word boundary
#                 space_pos = text.rfind(' ', start + min_len, end)
#                 if space_pos != -1:
#                     end = space_pos + 1
        
#         chunk = text[start:end].strip()
#         if chunk and len(chunk) >= min_len:
#             chunks.append(chunk)
        
#         # Move start position, considering overlap
#         start = end - overlap if (end - overlap) > start else end
#         if start >= n:
#             break
    
#     return chunks

def chunk_text(text: str, max_chars: int = 800, overlap: int = 150, min_len: int = 50) -> t.List[str]:
    """Better chunking with sentence awareness"""
    if not text or not text.strip():
        return []
    
    # Clean the text first
    text = re.sub(r'\s+', ' ', text.strip())
    
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        # Try to find a good break point
        end = start + max_chars
        
        # Don't break in the middle of a word if possible
        if end < n:
            # Look for natural break points
            for break_char in ['. ', '! ', '? ', '\n\n', '\n', '; ', ', ']:
                break_pos = text.rfind(break_char, start + min_len, end)
                if break_pos != -1:
                    end = break_pos + len(break_char.strip())
                    break
            else:
                # Fallback: break at word boundary
                space_pos = text.rfind(' ', start + min_len, end)
                if space_pos != -1:
                    end = space_pos + 1
        
        chunk = text[start:end].strip()
        if chunk and len(chunk) >= min_len:
            chunks.append(chunk)
        
        # Move start position, considering overlap
        start = end - overlap if (end - overlap) > start else end
        if start >= n:
            break
    
    return chunks

# --------------------
# Embeddings (provider-agnostic)
# --------------------
def sanitize_vector(vec: t.Optional[t.List[float]], dim: int) -> t.List[float]:
    if not isinstance(vec, list):
        return []
    v = [float(x) for x in vec if isinstance(x, (int, float))]
    if len(v) == 0:
        return []
    if len(v) == dim:
        return v
    if len(v) > dim:
        return v[:dim]
    return v + [0.0] * (dim - len(v))

def _embed_databricks(texts: t.List[str]) -> t.List[t.List[float]]:
    resp = dbx_client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": texts})
    data = resp.get("data", [])
    return [ (row.get("embedding") if isinstance(row, dict) else None) for row in data ]

def _embed_openai(texts: t.List[str]) -> t.List[t.List[float]]:
    # OpenAI returns in the request order
    resp = oa_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def embed_batch(texts: t.List[str], batch_size: int = 64, expected_dim: int = VECTOR_DIM) -> t.List[t.List[float]]:
    out: t.List[t.List[float]] = []
    if not texts:
        return out
    log_info("Embedding via", BACKEND_PROVIDER)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            raw = _embed_openai(batch) if BACKEND_PROVIDER == "OPENAI" else _embed_databricks(batch)
            if len(raw) != len(batch):
                log_warn(f"Embedding response size mismatch: expected {len(batch)}, got {len(raw)}")
            for r in raw:
                out.append(sanitize_vector(r, expected_dim))
        except Exception as e:
            log_error("Embedding batch failed:", e)
            out.extend([[] for _ in batch])
    return out

def embed_query_single(text: str) -> t.List[float]:
    try:
        raw = _embed_openai([text]) if BACKEND_PROVIDER == "OPENAI" else _embed_databricks([text])
        return sanitize_vector((raw[0] if raw else []), VECTOR_DIM)
    except Exception as e:
        log_error("Query embedding failed:", e)
        return []

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
            log_debug(f"Row: doc_id={r[0]} page={r[2]} chunk_id={r[3][:8]}.. content_len={len(r[4])} emb_len={len(r[5]) if r[5] else 0}")
        c.executemany(insert_sql, payload)
    conn.close()
    log_info(f"Inserted {len(rows)} rows into {DELTA_TABLE}")

def verify_embeddings(pdf_name: t.Optional[str] = None, limit: int = 5):
    conn = db_connect()
    with conn.cursor() as c:
        base = f"FROM {DELTA_TABLE}"
        params: t.Tuple = tuple()
        if pdf_name:
            base += " WHERE pdf_name = ?"
            params = (pdf_name,)

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



# ----------------------------
# PDF Ingestion
# ----------------------------
def ingest_pdf(pdf_input: t.Union[str, bytes], pdf_name: str, dedup: bool = True, chunk_size: int = 500, batch_size: int = 64):
    global INDEX, DOC_STORE
    
    import tempfile
    from PyPDF2 import PdfReader

    # create a single doc_id for this PDF
    doc_id = str(uuid.uuid4())

    # ----- accept bytes / file-like / path -----
    tmp_path = None
    created_tmp = False
    if isinstance(pdf_input, (bytes, bytearray)):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tf.write(pdf_input)
        tf.close()
        tmp_path = tf.name
        created_tmp = True
    elif hasattr(pdf_input, "read"):  # streamlit UploadedFile
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tf.write(pdf_input.read())
        tf.close()
        tmp_path = tf.name
        created_tmp = True
    elif isinstance(pdf_input, str) and os.path.exists(pdf_input):
        tmp_path = pdf_input
    else:
        raise ValueError("pdf_input must be bytes, file-like, or a valid file path")

    # Optional dedup: remove previous rows for same pdf_name
    if dedup:
        try:
            delete_existing_by_pdf_name(pdf_name)
        except Exception:
            log_warn("Could not delete existing rows for dedup. Continuing...")

    # Extract text with better method
    full_text = extract_text_from_pdf(tmp_path)
    
    if not full_text.strip():
        if created_tmp:
            try: os.remove(tmp_path)
            except: pass
        return {"doc_id": doc_id, "pages": 0, "chunks": 0}

    # Chunk the entire document text with improved chunking
    chunks = chunk_text(full_text, max_chars=chunk_size, overlap=150, min_len=50)
    
    total_chunks = len(chunks)
    if total_chunks == 0:
        if created_tmp:
            try: os.remove(tmp_path)
            except: pass
        return {"doc_id": doc_id, "pages": 0, "chunks": 0}

    # Get total pages for metadata
    reader = PdfReader(tmp_path)
    total_pages = len(reader.pages)

    # Batch-embed and prepare rows for write_rows
    rows_to_write: t.List[t.Tuple[str, str, int, str, str, t.List[float], dt.datetime]] = []
    faiss_vectors: t.List[np.ndarray] = []

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        # get embeddings (provider-agnostic)
        emb_batch = embed_batch(batch_chunks, batch_size=len(batch_chunks), expected_dim=VECTOR_DIM)

        for chunk_idx, (chunk_str, emb) in enumerate(zip(batch_chunks, emb_batch)):
            # Ensure we have a vector of length VECTOR_DIM. If service returned None/empty, replace with zeros.
            if not isinstance(emb, list) or len(emb) == 0:
                log_warn("Empty embedding received for a chunk — filling zeros to avoid null embeddings")
                emb = [0.0] * VECTOR_DIM
            
            # cast floats
            emb = [float(x) for x in emb]
            chunk_id = str(uuid.uuid4())
            created_at = dt.datetime.now(dt.timezone.utc)
            
            # Use page 1 for all chunks since we're chunking the whole document
            page_num = 1
            rows_to_write.append((doc_id, pdf_name, page_num, chunk_id, chunk_str, emb, created_at))

            # prepare normalized vector for FAISS / in-memory retrieval
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            else:
                # If vector has zero norm, use a small random vector to avoid issues
                vec = np.random.rand(VECTOR_DIM).astype(np.float32) * 0.001
            
            faiss_vectors.append(vec)

            # also append to in-memory DOC_STORE for immediate use
            doc_store_item = {
                "doc_id": doc_id,
                "pdf_name": pdf_name,
                "page": page_num,
                "chunk_id": chunk_id,
                "content": chunk_str,
                "embedding": vec.tolist()
            }
            DOC_STORE.append(doc_store_item)
            
            if DEBUG_VERBOSE:
                print(f"DOC_STORE size: {len(DOC_STORE)}")
                print(f"Added item - doc_id: {doc_store_item['doc_id']}")
                print(f"Added item - chunk_id: {doc_store_item['chunk_id']}")
                print(f"Added item - content: {doc_store_item['content'][:100]}...")

    # Write rows to Delta table in batches using existing helper
    batch_write_size = 256
    for i in range(0, len(rows_to_write), batch_write_size):
        write_rows(rows_to_write[i:i + batch_write_size])

    # Add all prepared vectors to FAISS (stacked add)
    if faiss_vectors:
        try:
            stack = np.vstack(faiss_vectors).astype("float32")
            INDEX.add(stack)
            print(f"DEBUG: Added {len(faiss_vectors)} vectors to FAISS index. Total index size: {INDEX.ntotal}")
        except Exception as e:
            log_error(f"Failed to add vectors to FAISS: {e}")
            # Recreate index
            INDEX = faiss.IndexFlatIP(VECTOR_DIM)
            if faiss_vectors:
                stack = np.vstack(faiss_vectors).astype("float32")
                INDEX.add(stack)
                print(f"DEBUG: Recreated FAISS index and added {len(faiss_vectors)} vectors")

    # Trigger index sync if needed (for delta-sync mode)
    trigger_index_sync_if_needed()

    # cleanup temp file if we created one
    if created_tmp and tmp_path and os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    print(f"Final DOC_STORE size: {len(DOC_STORE)}")
    print(f"Final INDEX size: {INDEX.ntotal}")
    return {"doc_id": doc_id, "pages": total_pages, "chunks": len(rows_to_write)}

# def ingest_pdf(pdf_input: t.Union[str, bytes], pdf_name: str, dedup: bool = True, chunk_size: int = 500, batch_size: int = 64):
#     """
#     Ingest a PDF (path or bytes or file-like), chunk it, embed in batches, write to Delta table,
#     populate in-memory FAISS index + DOC_STORE for immediate retrieval, and trigger index sync if needed.
#     Returns: {"doc_id": ..., "pages": <pages>, "chunks": <chunks_indexed>}
#     """
#     import tempfile
#     from PyPDF2 import PdfReader
#     global INDEX, DOC_STORE 

#     # create a single doc_id for this PDF
#     doc_id = str(uuid.uuid4())

#     # ----- accept bytes / file-like / path -----
#     tmp_path = None
#     created_tmp = False
#     if isinstance(pdf_input, (bytes, bytearray)):
#         tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#         tf.write(pdf_input)
#         tf.close()
#         tmp_path = tf.name
#         created_tmp = True
#     elif hasattr(pdf_input, "read"):  # streamlit UploadedFile
#         tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#         tf.write(pdf_input.read())
#         tf.close()
#         tmp_path = tf.name
#         created_tmp = True
#     elif isinstance(pdf_input, str) and os.path.exists(pdf_input):
#         tmp_path = pdf_input
#     else:
#         raise ValueError("pdf_input must be bytes, file-like, or a valid file path")

#     # Optional dedup: remove previous rows for same pdf_name
#     if dedup:
#         try:
#             delete_existing_by_pdf_name(pdf_name)
#         except Exception:
#             log_warn("Could not delete existing rows for dedup. Continuing...")

#     # Extract text with better method
#     full_text = extract_text_from_pdf(tmp_path)
    
#     if not full_text.strip():
#         if created_tmp:
#             try: os.remove(tmp_path)
#             except: pass
#         return {"doc_id": doc_id, "pages": 0, "chunks": 0}

#     # Chunk the entire document text with improved chunking
#     chunks = chunk_text(full_text, max_chars=chunk_size, overlap=150, min_len=50)
    
#     total_chunks = len(chunks)
#     if total_chunks == 0:
#         if created_tmp:
#             try: os.remove(tmp_path)
#             except: pass
#         return {"doc_id": doc_id, "pages": 0, "chunks": 0}

#     # Get total pages for metadata
#     reader = PdfReader(tmp_path)
#     total_pages = len(reader.pages)

#     # Batch-embed and prepare rows for write_rows
#     rows_to_write: t.List[t.Tuple[str, str, int, str, str, t.List[float], dt.datetime]] = []
#     faiss_vectors: t.List[np.ndarray] = []

#     for i in range(0, total_chunks, batch_size):
#         batch_chunks = chunks[i:i + batch_size]
#         # get embeddings (provider-agnostic)
#         emb_batch = embed_batch(batch_chunks, batch_size=len(batch_chunks), expected_dim=VECTOR_DIM)

#         for chunk_idx, (chunk_str, emb) in enumerate(zip(batch_chunks, emb_batch)):
#             # Ensure we have a vector of length VECTOR_DIM. If service returned None/empty, replace with zeros.
#             if not isinstance(emb, list) or len(emb) == 0:
#                 log_warn("Empty embedding received for a chunk — filling zeros to avoid null embeddings")
#                 emb = [0.0] * VECTOR_DIM
            
#             # cast floats
#             emb = [float(x) for x in emb]
#             chunk_id = str(uuid.uuid4())
#             created_at = dt.datetime.now(dt.timezone.utc)  # Use timezone-aware datetime
            
#             # Use page 1 for all chunks since we're chunking the whole document
#             page_num = 1
#             rows_to_write.append((doc_id, pdf_name, page_num, chunk_id, chunk_str, emb, created_at))

#             # prepare normalized vector for FAISS / in-memory retrieval
#             vec = np.array(emb, dtype=np.float32)
#             norm = np.linalg.norm(vec)
#             if norm > 0:
#                 vec = vec / norm
#             else:
#                 # If vector has zero norm, use a small random vector to avoid issues
#                 vec = np.random.rand(VECTOR_DIM).astype(np.float32) * 0.001
            
#             faiss_vectors.append(vec)

#             # also append to in-memory DOC_STORE for immediate use
#             doc_store_item = {
#                 "doc_id": doc_id,
#                 "pdf_name": pdf_name,
#                 "page": page_num,
#                 "chunk_id": chunk_id,
#                 "content": chunk_str,
#                 "embedding": vec.tolist()  # store normalized for scoring
#             }
#             DOC_STORE.append(doc_store_item)
            
#             if DEBUG_VERBOSE:
#                 print(f"DOC_STORE size: {len(DOC_STORE)}")
#                 print(f"DOC_STORE doc_id: {doc_store_item['doc_id']}")
#                 print(f"DOC_STORE chunk_id: {doc_store_item['chunk_id']}")
#                 print(f"DOC_STORE content preview: {doc_store_item['content'][:100]}...")
#                 print(f"DOC_STORE embedding length: {len(doc_store_item['embedding'])}")
#                 print(f"DOC_STORE object id: {id(DOC_STORE)}") 
#     # Write rows to Delta table in batches using existing helper
#     batch_write_size = 256
#     for i in range(0, len(rows_to_write), batch_write_size):
#         write_rows(rows_to_write[i:i + batch_write_size])

#     # Add all prepared vectors to FAISS (stacked add)
#     if faiss_vectors:
#         try:
#             stack = np.vstack(faiss_vectors).astype("float32")
#             INDEX.add(stack)
#             print(f"DEBUG: Added {len(faiss_vectors)} vectors to FAISS index. Total index size: {INDEX.ntotal}")
#         except Exception as e:
#             log_error(f"Failed to add vectors to FAISS: {e}")
#             # Try to recreate index if there's an issue
#             # Remove the global declaration since INDEX is already a global variable
#             # global INDEX
#             INDEX = faiss.IndexFlatIP(VECTOR_DIM)
#             if faiss_vectors:
#                 stack = np.vstack(faiss_vectors).astype("float32")
#                 INDEX.add(stack)
#                 print(f"DEBUG: Recreated FAISS index and added {len(faiss_vectors)} vectors")

#     # Trigger index sync if needed (for delta-sync mode)
#     trigger_index_sync_if_needed()

#     # cleanup temp file if we created one
#     if created_tmp and tmp_path and os.path.exists(tmp_path):
#         try:
#             os.remove(tmp_path)
#         except Exception:
#             pass

#     return {"doc_id": doc_id, "pages": total_pages, "chunks": len(rows_to_write)}




def extract_text_from_pdf(pdf_path: str) -> str:
    """Better PDF text extraction using pdfplumber"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Try multiple extraction strategies
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    # Fallback: extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            text += " ".join([str(cell) for cell in row if cell]) + "\n"
    except Exception as e:
        log_error(f"PDF extraction error: {e}")
        # Fallback to PyPDF2 if pdfplumber fails
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        except Exception as e2:
            log_error(f"PyPDF2 extraction also failed: {e2}")
    
    return text



def _question_context_overlap(question: str, ctx: str, ctx_emb: t.Optional[t.List[float]] = None) -> float:
    """
    Combine lexical overlap and provider semantic similarity (0..1).
    ctx_emb: optional precomputed embedding (normalized or raw). If not provided we call embed_query_single(ctx).
    """
    q_tokens = set(question.lower().split())
    c_tokens = set(ctx.lower().split())
    lexical_overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))

    # semantic similarity using provider embeddings
    try:
        q_emb = np.array(embed_query_single(question), dtype=np.float32)
        if ctx_emb is None:
            c_emb = np.array(embed_query_single(ctx), dtype=np.float32)
        else:
            c_emb = np.array(ctx_emb, dtype=np.float32)
        # if vectors are zero-length or empty -> sim = 0
        if q_emb.size == 0 or c_emb.size == 0 or np.linalg.norm(q_emb) == 0 or np.linalg.norm(c_emb) == 0:
            sim = 0.0
        else:
            sim = float(np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb)))
    except Exception as e:
        log_debug("semantic similarity failed:", e)
        sim = 0.0

    # balance lexical + semantic
    return 0.5 * lexical_overlap + 0.5 * sim


# --------------------
# Retrieval & Chat
# --------------------
def retrieve(question: str, k: int = 5):
    """Retrieve top-k chunks from in-memory FAISS + DOC_STORE"""
    global DOC_STORE, INDEX
    
    print(f"DEBUG: DOC_STORE object id: {id(DOC_STORE)}, length: {len(DOC_STORE)}")
    print(f"DEBUG: INDEX object id: {id(INDEX)}, ntotal: {INDEX.ntotal}")
    if len(DOC_STORE) == 0:
        print("DEBUG: DOC_STORE is empty")
        try:
            from __main__ import DOC_STORE as main_doc_store
            print(f"DEBUG: Main DOC_STORE length: {len(main_doc_store)}")
            DOC_STORE = main_doc_store
        except:
            pass        
        return []

    q_emb = embed_query_single(question)
    if not q_emb or len(q_emb) == 0:
        print("DEBUG: Query embedding failed or is empty")
        return []

    q_vec = np.array(q_emb, dtype=np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec = q_vec / norm
    else:
        print("DEBUG: Query vector has zero norm")
        return []

    print(f"DEBUG: FAISS index size: {INDEX.ntotal}, DOC_STORE size: {len(DOC_STORE)}")
    
    # Check if index has vectors
    if INDEX.ntotal == 0:
        print("DEBUG: FAISS index is empty")
        return []

    try:
        D, I = INDEX.search(np.array([q_vec], dtype=np.float32), min(k, INDEX.ntotal))
        print(f"DEBUG: Search results - Distances: {D}, Indices: {I}")
        
        retrieved = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1 or idx >= len(DOC_STORE):
                continue
            ctx = DOC_STORE[int(idx)]
            # Use a simpler scoring approach for now
            retrieved.append({**ctx, "score": float(score)})
        
        retrieved = sorted(retrieved, key=lambda x: x["score"], reverse=True)
        print(f"DEBUG: Retrieved {len(retrieved)} chunks")
        return retrieved[:k]
        
    except Exception as e:
        print(f"DEBUG: FAISS search failed: {e}")
        return []


def _chat_databricks(messages: t.List[dict], max_tokens: int) -> str:
    resp = dbx_client.predict(endpoint=CHAT_ENDPOINT, inputs={"messages": messages, "max_tokens": max_tokens})
    if "choices" in resp:
        return resp["choices"][0]["message"]["content"]
    if "content" in resp:
        return resp["content"]
    return str(resp)[:2000]

def _chat_openai(messages: t.List[dict], max_tokens: int) -> str:
    # Convert roles if needed: we already use OpenAI-compatible roles
    resp = oa_client.chat.completions.create(model=OPENAI_CHAT_MODEL, messages=messages, max_tokens=max_tokens)
    return resp.choices[0].message.content

def call_llm(messages: t.List[dict], max_tokens: int = 400) -> str:
    try:
        return _chat_openai(messages, max_tokens) if BACKEND_PROVIDER == "OPENAI" else _chat_databricks(messages, max_tokens)
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

    msgs = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": question}]
    reply = call_llm(msgs, max_tokens=max_tokens)
    return reply, ctx_list

# ----------------------------
# Chat with RAG
# ----------------------------
def chat_with_rag(
    question: str,
    history: t.Optional[t.List[dict]] = None,
    k: int = 8,
    max_tokens: int = 512
):
    """
    Strict RAG pipeline. Only uses retrieved context. 
    If no context is relevant, replies "I don't know".
    """
    history = history or []
    
    # Retrieve context
    ctx_list = retrieve(question, k=k)


    # DEBUG: Show what was retrieved
    print(f"Retrieved {len(ctx_list)} chunks for question: '{question}'")
    for i, ctx in enumerate(ctx_list):
        print(f"Chunk {i+1}: {ctx['content'][:200]}...")

    # Build strict system prompt
    if ctx_list:
        context_text = "\n\n".join(
            f"[{c['pdf_name']} - p.{c['page']}]: {c['content']}"
            for c in ctx_list
        )
        system_prompt = (
            "You are a helpful document analysis assistant.\n"
            "Use the retrieved context below to answer the user's question accurately.\n"
            "If the context contains the information needed to answer, provide a clear answer.\n"
            "If the context is insufficient, you may say you don't know, but first try to infer from available information.\n\n"
            f"Retrieved context:\n{context_text}"
        )
    else:
        system_prompt = (
            "You are a helpful assistant. No relevant context was found for this question."
        )

    # Build message history
    msgs = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": question}
    ]

    # Call the LLM
    reply = call_llm(msgs, max_tokens=max_tokens)

    # Post-check: enforce "I don't know" if no context
    if (not ctx_list) and reply.lower() not in ["i don't know", "i dont know"]:
        reply = "I don't know"

    return reply, ctx_list


def load_existing_into_memory(limit: t.Optional[int] = None):
    """
    Load existing rows from DELTA_TABLE into DOC_STORE + FAISS.
    Use only for small collections (dev); large tables will OOM.
    """
    conn = db_connect()
    with conn.cursor() as c:
        q = f"SELECT chunk_id, content, embedding FROM {DELTA_TABLE}"
        if limit:
            q += f" LIMIT {limit}"
        c.execute(q)
        rows = c.fetchall()
    conn.close()

    to_add = []
    for r in rows:
        chunk_id, content, emb = r[0], r[1], r[2]
        if not emb or len(emb) == 0:
            continue
        vec = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            continue
        DOC_STORE.append({
            "doc_id": None,
            "pdf_name": None,
            "page": None,
            "chunk_id": chunk_id,
            "content": content,
            "embedding": vec.tolist()
        })
        to_add.append(vec)
    if to_add:
        INDEX.add(np.vstack(to_add).astype("float32"))


# --------------------
# Optional post-ingestion hook
# --------------------
def on_ingest(result: dict):
    # You can extend this to auto-summarize the PDF, write metadata, etc.
    log_debug("on_ingest called with:", result)

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Databricks Agentic RAG", page_icon="🧠")
st.title("🧠 Databricks Agentic RAG Chat — Robust Embeddings")

# Probe embedding dimension once
try:
    if BACKEND_PROVIDER == "OPENAI":
        vec = _embed_openai(["_probe_"])
        dim_probe = len(vec[0]) if vec and vec[0] else 0
    else:
        resp_test = dbx_client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": ["_probe_"]})
        dim_probe = len((resp_test.get("data", [{}])[0] or {}).get("embedding", []) or [])
    if dim_probe:
        VECTOR_DIM = dim_probe
        st.caption(f"Embedding endpoint OK • dimension={VECTOR_DIM} • provider={BACKEND_PROVIDER}")
    else:
        st.caption(f"Embedding probe returned no vector — using default {EMBED_DIM_DEFAULT}.")
except Exception as e:
    st.warning(f"Embedding endpoint probe failed: {e}")

# Ensure resources at startup
ensure_table_and_index(dim=VECTOR_DIM or EMBED_DIM_DEFAULT)

with st.sidebar:
    st.header("Upload PDF")
    f = st.file_uploader("Choose a PDF", type=["pdf"])
    dedup = st.checkbox("De-duplicate rows for same PDF name before ingest", value=True)

    if st.button("Ingest", disabled=not f):
        file_bytes = f.read()
        r = ingest_pdf(file_bytes, f.name, dedup=dedup)
        query = "insurance certificate details"
        emb = embed_batch([query], batch_size=1, expected_dim=VECTOR_DIM)[0]

        # normalize query embedding
        import numpy as np
        vec = np.array(emb, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)

        # search top-3 from FAISS
        D, I = INDEX.search(np.expand_dims(vec, axis=0), 3)

        print("Top-3 similarity scores:", D)
        for idx in I[0]:
            if idx < len(DOC_STORE):
                hit = DOC_STORE[idx]
                print(f"Page {hit['page']} → {hit['content'][:200]}...")

        if INDEX_MODE == "SYNC":
            st.success(f"Ingested {r['chunks']} chunks (doc_id={r['doc_id']}) and index sync triggered ✅")
        else:
            st.success(f"Ingested {r['chunks']} chunks (doc_id={r['doc_id']}) ✅ (Direct Access)")
        on_ingest(r)

    st.divider()
    st.subheader("Index Info")
    try:
        idx_obj = vsc.get_index(VS_ENDPOINT, VS_INDEX)
        idx_meta = {
            "endpoint": VS_ENDPOINT,
            "index": VS_INDEX,
            "status": getattr(idx_obj, "status", "?"),
            "pipeline_type": getattr(idx_obj, "pipeline_type", "DIRECT_ACCESS"),
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
            ans, ctx = chat_with_rag(q, st.session_state.history[:-1], k=8)
        st.markdown(ans)
        if ctx:
            with st.expander("Sources"):
                for c in ctx:
                    st.markdown(f"📄 **{c['pdf_name']}** (p.{c['page']})\n\n> {c['content'][:400]}…")
    st.session_state.history.append({"role": "assistant", "content": ans})


# import logging
# import os
# import streamlit as st
# import pdfplumber
# from model_serving_utils import query_endpoint, is_endpoint_supported
# from databricks.sdk import WorkspaceClient
# from dotenv import load_dotenv
# load_dotenv()
# from fastapi import FastAPI
# from databricks.vector_search.client import VectorSearchClient
# # Read from environment variables (Streamlit secrets)

# app = FastAPI()

# DBX_HOST = os.getenv("DATABRICKS_HOST")
# DBX_TOKEN = os.getenv("DATABRICKS_TOKEN")
# SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT")
# CHAT_ENDPOINT = os.getenv("CHAT_ENDPOINT")
# EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")

# # Initialize Databricks WorkspaceClient
# w = WorkspaceClient(host=DBX_HOST, token=DBX_TOKEN)

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Ensure environment variable is set correctly
# SERVING_ENDPOINT = "databricks-llama-4-maverick"
# # os.getenv('SERVING_ENDPOINT')
# assert SERVING_ENDPOINT, (
#     "Unable to determine serving endpoint to use for chatbot app. If developing locally, "
#     "set the SERVING_ENDPOINT environment variable to the name of your serving endpoint. If "
#     "deploying to a Databricks app, include a serving endpoint resource named "
#     "'serving_endpoint' with CAN_QUERY permissions."
# )

# # Check if endpoint is supported
# endpoint_supported = is_endpoint_supported(SERVING_ENDPOINT)

# def get_user_info():
#     headers = st.context.headers
#     return dict(
#         user_name=headers.get("X-Forwarded-Preferred-Username"),
#         user_email=headers.get("X-Forwarded-Email"),
#         user_id=headers.get("X-Forwarded-User"),
#     )

# user_info = get_user_info()

# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "system", "content": "You are a helpful assistant in a Databricks app."}
#     ]
# if "pdf_content" not in st.session_state:
#     st.session_state.pdf_content = ""

# st.title("🧱 Conversational AI Chatbot with PDF Support")

# uploaded_file = st.file_uploader("📄 Upload a PDF for context", type=["pdf"])
# if uploaded_file:
#     with pdfplumber.open(uploaded_file) as pdf:
#         pdf_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
#     st.session_state.pdf_content = pdf_text.strip()
#     st.success("✅ PDF uploaded and processed successfully!")

# if not endpoint_supported:
#     st.error("⚠️ Unsupported Endpoint Type")
#     st.markdown(
#         f"The endpoint `{SERVING_ENDPOINT}` is not compatible with this chatbot."
#     )
# else:
#     st.markdown("💬 Chat below. Your history will persist for the session.")

#     for message in st.session_state.messages:
#         if message["role"] != "system"
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#     if prompt := st.chat_input("Type your message"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         messages_to_send = st.session_state.messages.copy()
#         if st.session_state.pdf_content:
#             messages_to_send.insert(1, {
#                 "role": "system",
#                 "content": f"The user has uploaded a PDF. Here is its content:\n{st.session_state.pdf_content}"
#             })

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 assistant_reply = query_endpoint(
#                     endpoint_name=SERVING_ENDPOINT,
#                     messages=messages_to_send,
#                     max_tokens=400,
#                 )["content"]
#                 st.markdown(assistant_reply)

#         st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# vsc = VectorSearchClient(disable_notice=True)
# VS_ENDPOINT = "vs_endpoint_default"
# VS_INDEX = "main.default.pdf_text_managed_vs_index"

# @app.post("/search")
# def search_docs(query: str):
#     results = vsc.get_index(VS_ENDPOINT, VS_INDEX).similarity_search(
#         query_text=query,
#         num_results=5,
#         columns=["pdf_name", "content"]
#     )
#     return results