import os, io, uuid, datetime as dt, json
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from databricks import sql
from databricks.vector_search.client import VectorSearchClient
from mlflow.deployments import get_deploy_client

# --------------------
# Load environment variables
# --------------------
load_dotenv()
CHAT_ENDPOINT = os.getenv("CHAT_ENDPOINT")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")
VS_ENDPOINT = os.getenv("VS_ENDPOINT")
VS_INDEX = os.getenv("VS_INDEX")
DELTA_TABLE = os.getenv("DELTA_TABLE")
AGENTIC_MODE = os.getenv("AGENTIC_MODE", "False").lower() == "true"

DB_HOST = os.getenv("DATABRICKS_HOST", "").replace("https://", "")
DB_HTTP_PATH = os.getenv("DATABRICKS_SQL_HTTP_PATH")
DB_TOKEN = os.getenv("DATABRICKS_TOKEN")

# --------------------
# Databricks clients
# --------------------
client = get_deploy_client("databricks")
vsc = VectorSearchClient()

# --------------------
# Utility: Delta table & VS index setup
# --------------------
def ensure_table_and_index(dim: int = 1024):
    """Ensure Delta table, VS endpoint, and index exist."""
    
    # --- Delta table ---
    conn = sql.connect(server_hostname=DB_HOST, http_path=DB_HTTP_PATH, access_token=DB_TOKEN)
    with conn.cursor() as c:
        c.execute(f"""
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
        """)
    conn.close()
    
    # --- Vector Search endpoint ---
    endpoints = vsc.list_endpoints().get("endpoints", [])
    if endpoints:
        existing = endpoints[0]["name"]
        st.info(f"Using existing VS endpoint: {existing}")
        os.environ["VS_ENDPOINT"] = existing
    else:
        ep = vsc.create_endpoint(name="rag-vs-endpoint", endpoint_type="STANDARD")
        st.info(f"Created VS endpoint: {ep['name']}")
        os.environ["VS_ENDPOINT"] = ep["name"]
    
    # --- Index ---
    try:
        idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)
        st.info(f"Index exists: {VS_INDEX}, reusing ✅")
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
            st.warning(f"Index {VS_INDEX} not found. Creating...")
            vsc.create_delta_sync_index(
                endpoint_name=VS_ENDPOINT,
                index_name=VS_INDEX,
                source_table_name=DELTA_TABLE,
                pipeline_type="TRIGGERED",
                primary_key="chunk_id",
                embedding_dimension=dim,
                embedding_vector_column="embedding"
            )
            st.success(f"Created index {VS_INDEX} ✅")
        else:
            raise

def sync_index_safe():
    """Trigger sync if TRIGGERED/CONTINUOUS; skip Direct Access."""
    try:
        idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)
        pipeline_type = idx.get("pipeline_type", "DIRECT_ACCESS").upper()
        if pipeline_type in ("TRIGGERED", "CONTINUOUS"):
            vsc.trigger_index_sync(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
            st.success(f"Index sync triggered ✅ (type={pipeline_type})")
        else:
            st.info(f"Direct Access index → no sync needed (type={pipeline_type})")
    except Exception as e:
        st.error(f"Failed to sync index: {e}")

# --------------------
# PDF ingestion & chunking
# --------------------
def chunk_text(text, max_chars=1200, overlap=200):
    """Split text into overlapping chunks."""
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        k = text.rfind(" ", i + max_chars - overlap, j)
        j = k if k != -1 else j
        chunks.append(text[i:j])
        i = max(j - overlap, j)
    return [c for c in chunks if c.strip()]

def embed_chunks(chunks):
    """Embed text chunks and safely handle empty embeddings."""
    if not chunks:
        st.warning("⚠️ No chunks to embed.")
        return []

    try:
        out = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": chunks})
        data = out.get("data", [])
        embeddings = []
        for idx, row in enumerate(data):
            emb = row.get("embedding", [])
            if not emb:
                st.warning(f"⚠️ Chunk {idx} failed to embed: {chunks[idx][:100]}")
            embeddings.append(emb or [])
        return embeddings
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return [[] for _ in chunks]

def ingest_pdf(file_bytes, pdf_name):
    """Full ingestion: PDF → chunks → embeddings → Delta table."""
    doc_id = str(uuid.uuid4())
    rows = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        st.info(f"Processing PDF: {pdf_name} (pages={len(pdf.pages)})")
        for pageno, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                st.warning(f"Page {pageno} has no text. Skipping.")
                continue

            chunks = chunk_text(text)
            embs = embed_chunks(chunks)

            now = dt.datetime.now(dt.timezone.utc)
            for ch, emb in zip(chunks, embs):
                rows.append((doc_id, pdf_name, pageno, str(uuid.uuid4()), ch, emb, now))

    if not rows:
        st.error(f"No valid chunks found for {pdf_name}")
        return {"doc_id": doc_id, "chunks": 0}

    write_rows(rows)
    st.success(f"Ingested {len(rows)} chunks from {pdf_name}")
    verify_embeddings(pdf_name)
    sync_index_safe()
    return {"doc_id": doc_id, "chunks": len(rows)}

def write_rows(rows):
    """Write chunked PDF + embeddings to Delta table."""
    conn = sql.connect(server_hostname=DB_HOST, http_path=DB_HTTP_PATH, access_token=DB_TOKEN)
    with conn.cursor() as c:
        for r in rows:
            emb_json = json.dumps(r[5]) if r[5] else None
            c.execute(f"""INSERT INTO {DELTA_TABLE} 
                (doc_id, pdf_name, page, chunk_id, content, embedding, created_at) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (r[0], r[1], r[2], r[3], r[4], emb_json, r[6])
            )
    conn.close()

def verify_embeddings(pdf_name=None, limit=5):
    """Check for missing embeddings in Delta table."""
    conn = sql.connect(server_hostname=DB_HOST, http_path=DB_HTTP_PATH, access_token=DB_TOKEN)
    with conn.cursor() as c:
        query = f"SELECT pdf_name, page, chunk_id, embedding, content FROM {DELTA_TABLE}"
        if pdf_name:
            query += " WHERE pdf_name = ?"
            c.execute(query, (pdf_name,))
        else:
            c.execute(query)
        rows = c.fetchall()

    total, missing = len(rows), 0
    st.info(f"Verifying embeddings ({pdf_name or 'ALL PDFs'})")
    for r in rows[:limit]:
        emb_len = len(r[3]) if r[3] else 0
        status = "✅" if emb_len > 0 else "⚠️ MISSING"
        st.write(f"{r[0]} p.{r[1]} chunk {r[2]} → emb_len={emb_len} {status}")
        st.write(f"Preview: {r[4][:100]}...\n")
        if emb_len == 0: missing += 1

    for r in rows:
        if not r[3] or len(r[3]) == 0: missing += 1
    st.success(f"Total chunks={total}, missing embeddings={missing}")
    conn.close()

# --------------------
# Retrieval & RAG
# --------------------
def embed_query_single(text):
    """Return embedding vector for a single query."""
    out = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": [text]})
    data = out.get("data", [])
    return data[0].get("embedding", []) if data else []

def retrieve(query: str, k: int = 5):
    """Retrieve top-k chunks relevant to the query."""
    try:
        idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)
        try:
            res = idx.similarity_search(query_text=query, num_results=k, columns=["doc_id","pdf_name","page","content"])
        except Exception as e:
            if "query vector must be specified" in str(e):
                q_emb = embed_query_single(query)
                res = idx.similarity_search(query_vector=q_emb, num_results=k, columns=["doc_id","pdf_name","page","content"])
            else:
                raise
        matches = res.get("result", {}).get("data_array", [])
        return [{"doc_id": m[0], "pdf_name": m[1], "page": m[2], "content": m[3]} for m in matches]
    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        return []

def call_llm(messages, max_tokens=400):
    """Call Databricks LLM endpoint."""
    resp = client.predict(endpoint=CHAT_ENDPOINT, inputs={"messages": messages, "max_tokens": max_tokens})
    if "choices" in resp: return resp["choices"][0]["message"]["content"]
    if "content" in resp: return resp["content"]
    return str(resp)[:2000]

def agentic_chat(question: str, history: list, k=5, max_tokens=512):
    """Agentic RAG chat."""
    ctx_list = retrieve(question, k=k)
    if ctx_list:
        context_text = "\n\n".join(f"[{c['pdf_name']} p.{c['page']}]: {c['content']}" for c in ctx_list)
        system_prompt = f"You are a helpful assistant. Use retrieved context.\n\n{context_text}"
    else:
        system_prompt = "You are a helpful assistant. No context found."

    msgs = [{"role":"system","content":system_prompt}] + history + [{"role":"user","content":question}]
    reply = call_llm(msgs, max_tokens=max_tokens)
    return reply, ctx_list

def chat_with_rag(question, history=None, k=5, max_tokens=512):
    """Wrapper for Agentic or simple RAG mode."""
    history = history or []
    if AGENTIC_MODE:
        return agentic_chat(question, history, k=k, max_tokens=max_tokens)

    ctx_list = retrieve(question, k=k)
    context_text = "\n\n".join(f"[{c['pdf_name']} p.{c['page']}]: {c['content']}" for c in ctx_list) if ctx_list else ""
    system_prompt = f"Answer using ONLY this context.\n{context_text}" if ctx_list else "Answer using ONLY this context. No context found."
    msgs = [{"role":"system","content":system_prompt}] + history + [{"role":"user","content":question}]
    reply = call_llm(msgs, max_tokens=max_tokens)
    return reply, ctx_list

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Databricks Agentic RAG", page_icon="🧠")
st.title("🧠 Databricks Agentic RAG Chat")

ensure_table_and_index()

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])
if st.sidebar.button("Ingest", disabled=not uploaded_file):
    result = ingest_pdf(uploaded_file.read(), uploaded_file.name)
    st.sidebar.success(f"Ingested {result['chunks']} chunks (doc_id={result['doc_id']})")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask me anything…")
if query:
    st.session_state.history.append({"role":"user","content":query})
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..." if AGENTIC_MODE else "🤔 Thinking…"):
            answer, sources = chat_with_rag(query, st.session_state.history[:-1])
        st.markdown(answer)
        with st.expander("Sources"):
            for src in sources:
                st.markdown(f"📄 **{src['pdf_name']}** (p.{src['page']}): {src['content'][:300]}...")
    st.session_state.history.append({"role":"assistant","content":answer})