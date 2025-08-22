import os, io, uuid, datetime as dt
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from databricks import sql
from databricks.vector_search.client import VectorSearchClient
from mlflow.deployments import get_deploy_client
import json

# --------------------
# Load ENV
# --------------------
load_dotenv()
CHAT_ENDPOINT = os.getenv("CHAT_ENDPOINT")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")
VS_ENDPOINT = os.getenv("VS_ENDPOINT")
VS_INDEX = os.getenv("VS_INDEX")
DELTA_TABLE = os.getenv("DELTA_TABLE")
AGENTIC_MODE = os.getenv("AGENTIC_MODE","False").lower() == "true"

DB_HOST = os.getenv("DATABRICKS_HOST","").replace("https://","")
DB_HTTP_PATH = os.getenv("DATABRICKS_SQL_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

client = get_deploy_client("databricks")
vsc = VectorSearchClient()

# --------------------
# Ensure Delta table + VS index
# --------------------
def ensure_table_and_index(dim: int = 1024):
    """Ensure Delta table, VS endpoint, and sync index are ready for RAG."""

    # 1. Ensure table exists with CDF enabled
    conn = sql.connect(
        server_hostname=DB_HOST,
        http_path=DB_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
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

    # 2. Ensure endpoint (reuse existing if one exists)
    eps = vsc.list_endpoints().get("endpoints", [])
    if eps:
        existing = eps[0]["name"]
        st.info(f"Using existing VS endpoint: {existing}")
        os.environ["VS_ENDPOINT"] = existing
        globals()["VS_ENDPOINT"] = existing
    else:
        ep = vsc.create_endpoint(name="rag-vs-endpoint", endpoint_type="STANDARD")
        st.info(f"Created new VS endpoint: {ep['name']}")
        os.environ["VS_ENDPOINT"] = ep["name"]
        globals()["VS_ENDPOINT"] = ep["name"]

    # 3. Ensure index (Direct Access is fine here)
    try:
        idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)
        st.info(f"Index already exists: {VS_INDEX}, reusing it ✅")
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
            st.warning(f"Index {VS_INDEX} not found, creating it...")
            vsc.create_delta_sync_index(
                endpoint_name=VS_ENDPOINT,
                index_name=VS_INDEX,
                source_table_name=DELTA_TABLE,
                pipeline_type="TRIGGERED",   # CONTINUOUS not supported in your workspace
                primary_key="chunk_id",
                embedding_dimension=dim,
                embedding_vector_column="embedding"
                # ❌ no embedding_model_endpoint_name → Direct Access index
            )
            st.success(f"Created index {VS_INDEX} as Direct Access ✅")
        else:
            raise

def sync_index_safe():
    """
    Trigger sync on TRIGGERED or CONTINUOUS pipeline indexes after new data is added.
    Skips syncing for Direct Access indexes (immediate queryable).
    """
    try:
        st.info("Checking if index needs sync...")
        idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)

        # Check the pipeline type of the index
        pipeline_type = idx.get("pipeline_type", "DIRECT_ACCESS").upper()
        if pipeline_type in ("TRIGGERED", "CONTINUOUS"):
            # Trigger sync via the client
            vsc.trigger_index_sync(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
            st.success(f"Index sync triggered ✅ (pipeline_type={pipeline_type})")
        else:
            st.info(f"Direct Access index → no sync needed (pipeline_type={pipeline_type})")
    except Exception as e:
        st.error(f"Failed to sync index: {e}")



# --------------------
# PDF ingestion
# --------------------
def chunk_text(text, max_chars=1200, overlap=200):
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        j = min(i+max_chars, len(text))
        k = text.rfind(" ", i+max_chars-200, j)
        j = k if k!=-1 else j
        chunks.append(text[i:j])
        i = max(j-overlap, j)
    return [c for c in chunks if c.strip()]

def embed_chunks(chunks):
    """Embed a list of text chunks with debug logging and validation."""
    if not chunks:
        print("⚠️ No chunks provided to embed.")
        return []

    try:
        st.write("DEBUG: EMBEDDING_ENDPOINT →", EMBEDDING_ENDPOINT)
        st.write("DEBUG: first chunk to embed:", chunks[0][:50])

        out = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": chunks})
        data = out.get("data", [])
        embeddings = []
        for idx, row in enumerate(data):
            emb = row.get("embedding")
            st.write(f"Chunk {idx} embedding length:", len(emb) if emb else 0)
            if emb is None or len(emb) == 0:
                print(f"⚠️ Embedding missing for chunk {idx}: {chunks[idx][:50]}...")
                st.warning(f"⚠️ Chunk {idx} failed to embed. Text preview: {chunks[idx][:100]}...")
            embeddings.append(emb or [])
        return embeddings
    except Exception as e:
        print("❌ Error during embedding:", e)
        return [[] for _ in chunks]


def ingest_pdf(file_bytes, pdf_name):
    """Ingest PDF, chunk text, generate embeddings, write to Delta table, and verify embeddings."""
    import pdfplumber, uuid, datetime as dt
    doc_id = str(uuid.uuid4())
    rows = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        st.info(f"ℹ️ Processing PDF: {pdf_name}, total pages={len(pdf.pages)}")
        for pageno, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                st.warning(f"⚠️ Page {pageno} has no extractable text. Skipping.")
                continue

            chunks = chunk_text(text)
            if not chunks:
                st.warning(f"⚠️ Page {pageno}: no chunks generated after splitting.")
                continue

            embs = embed_chunks(chunks)
            if not embs or all(len(e) == 0 for e in embs):
                st.warning(f"⚠️ Page {pageno}: all embeddings missing for {len(chunks)} chunks.")

            now = dt.datetime.now(dt.timezone.utc)
            for ch, emb in zip(chunks, embs):
                rows.append((doc_id, pdf_name, pageno, str(uuid.uuid4()), ch, emb, now))
    
    if not rows:
        st.error(f"❌ No rows to write for PDF: {pdf_name}")
        return {"doc_id": doc_id, "chunks": 0}

    # Write to Delta table
    write_rows(rows)
    st.success(f"✅ PDF ingested: {pdf_name}, total chunks={len(rows)}")

    # Verify embeddings immediately after ingestion
    verify_embeddings(pdf_name)

    return {"doc_id": doc_id, "chunks": len(rows)}



def write_rows(rows):
    """Write chunked PDF + embeddings to Delta table safely."""
    conn = sql.connect(server_hostname=DB_HOST, http_path=DB_HTTP_PATH, access_token=DATABRICKS_TOKEN)
    with conn.cursor() as c:
        for r in rows:
            # r[5] is embedding: should be list[float] or empty list
            emb_list = r[5] if r[5] else []
            c.execute(f"""INSERT INTO {DELTA_TABLE} 
                (doc_id, pdf_name, page, chunk_id, content, embedding, created_at) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (r[0], r[1], r[2], r[3], r[4], emb_list, r[6])
            )
    conn.close()

    



def verify_embeddings(pdf_name=None, limit=5):
    """
    Verify embeddings in the Delta table using Python.
    Checks for empty arrays or NULL embeddings.
    Prints top rows and summary of missing embeddings.
    """
    conn = sql.connect(server_hostname=DB_HOST, http_path=DB_HTTP_PATH, access_token=DATABRICKS_TOKEN)
    with conn.cursor() as c:
        query = f"SELECT pdf_name, page, chunk_id, embedding, content FROM {DELTA_TABLE}"
        if pdf_name:
            query += f" WHERE pdf_name = ?"
            c.execute(query, (pdf_name,))
        else:
            c.execute(query)
        rows = c.fetchall()

    total = len(rows)
    missing = 0
    st.info(f"🔍 Verifying embeddings for PDF: {pdf_name or 'ALL PDFs'}")
    for r in rows[:limit]:
        emb = r[3]
        emb_len = len(emb) if emb else 0
        status = "✅" if emb_len > 0 else "⚠️ MISSING"
        st.write(f"PDF={r[0]}, Page={r[1]}, Chunk={r[2]}, Embedding length={emb_len} {status}")
        st.write(f"Content preview: {r[4][:100]}...\n")
        if emb_len == 0:
            missing += 1

    # Count all missing embeddings
    for r in rows:
        emb = r[3]
        if not emb or len(emb) == 0:
            missing += 1

    st.success(f"Total chunks: {total}, Chunks missing embeddings: {missing}")
    conn.close()

#------------------
# Agentic Retriever
# --------------------
def retrieve(query, k=5):
    idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)

    # ✅ manually embed query because this is a Direct Access index
    q_emb = client.predict(
        endpoint=EMBEDDING_ENDPOINT,
        inputs={"input": [query]}
    )["data"][0]["embedding"]

    res = idx.similarity_search(
        query_vector=q_emb,  # ✅ use query_vector, not query_text
        num_results=k,
        columns=["doc_id","pdf_name","page","content"]
    )

    rows = res.get("result",{}).get("data_array",[])
    return [{"doc_id":r[0],"pdf_name":r[1],"page":r[2],"content":r[3]} for r in rows]

def call_llm(messages, max_tokens=400):
    resp = client.predict(endpoint=CHAT_ENDPOINT, inputs={"messages": messages, "max_tokens": max_tokens})
    if "choices" in resp: return resp["choices"][0]["message"]["content"]
    if "content" in resp: return resp["content"]
    return str(resp)[:2000]

# ---- Replace/insert these helper functions ----
# ---- Replace/insert these helper functions ----
def embed_query_single(text):
    """
    Return a single embedding vector for `text` using the EMBEDDING_ENDPOINT.
    Reuses the databricks deployment client.
    """
    # The embedding endpoint in your setup expects an array-of-inputs or a single input.
    # We'll call it with a single-item list to get a single embedding back.
    out = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": [text]})
    # expected shape: out.get("data", []) -> list of rows, each a dict with "embedding"
    data = out.get("data", [])
    if not data:
        raise RuntimeError(f"Embedding endpoint returned no data: {out}")
    return data[0].get("embedding", [])


def retrieve(query: str, k: int = 5):
    """
    Retrieve top-k chunks relevant to the query from Vector Search.
    Returns: list of dicts: {"doc_id":..., "pdf_name":..., "page":..., "content":...}
    Works for both text-query-capable indexes and Direct Access indexes (by falling back
    to query_vector).
    """
    try:
        idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)

        # First try text-based query (works if the index supports query_text)
        try:
            res = idx.similarity_search(
                query_text=query,
                num_results=k,
                columns=["doc_id", "pdf_name", "page", "content"]
            )
        except Exception as e:
            # If the index requires a query vector (Direct Access), compute embedding and retry
            err_text = str(e)
            if "query vector must be specified" in err_text or "RESOURCE_DOES_NOT_EXIST" in err_text or "INVALID_PARAMETER_VALUE" in err_text:
                # compute vector and retry
                emb = embed_query_single(query)
                res = idx.similarity_search(
                    query_vector=emb,
                    num_results=k,
                    columns=["doc_id", "pdf_name", "page", "content"]
                )
            else:
                # re-raise if it's some other error
                raise

        matches = res.get("result", {}).get("data_array", [])
        out = []
        for m in matches:
            # m format: [doc_id, pdf_name, page, content]
            # safety: some entries may be None or shorter - guard against that
            doc_id = m[0] if len(m) > 0 else None
            pdf_name = m[1] if len(m) > 1 else "unknown"
            page = m[2] if len(m) > 2 else None
            content = m[3] if len(m) > 3 else ""
            out.append({
                "doc_id": doc_id,
                "pdf_name": pdf_name,
                "page": page,
                "content": content
            })
        return out

    except Exception as e:
        st.error(f"⚠️ Retrieval failed: {e}")
        return []


# ---- Small adjustments to agentic_chat and chat_with_rag to consume retrieve()'s list ----
def agentic_chat(question: str, history: list, k: int = 5, max_tokens: int = 512):
    """
    Answer user query using retrieval-augmented generation (RAG).
    Works with Streamlit history format: {"role":"user"/"assistant","content":...}
    """

    # Step 1: Retrieve relevant context
    ctx_list = retrieve(question, k=k)

    # Step 2: Build system prompt including retrieved context
    if ctx_list:
        context_text = "\n\n".join(f"[{c['pdf_name']} - page {c['page']}]: {c['content']}" for c in ctx_list)
        system_prompt = (
            "You are a helpful assistant. Use the retrieved context from the uploaded document "
            "to answer the user's question. If the context is insufficient, say so.\n\n"
            f"Retrieved context:\n{context_text}"
        )
    else:
        system_prompt = "You are a helpful assistant. No relevant document context was found."

    # Step 3: Convert Streamlit history to LLM message format
    msgs = [{"role": "system", "content": system_prompt}]
    for h in history:
        msgs.append({"role": h["role"], "content": h["content"]})

    # Step 4: Append current question
    msgs.append({"role": "user", "content": question})

    # Step 5: Call Databricks LLM endpoint
    resp = client.predict(endpoint=CHAT_ENDPOINT, inputs={"messages": msgs, "max_tokens": max_tokens})
    reply = resp.get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No reply generated.")
    return reply, ctx_list


def chat_with_rag(question, history=None, k=5, max_tokens=512):
    """
    Wrapper that calls agentic_chat if AGENTIC_MODE else a simple RAG path.
    Works with Streamlit history format: {"role":"user"/"assistant","content":...}
    """
    history = history or []

    if AGENTIC_MODE:
        return agentic_chat(question, history, k=k, max_tokens=max_tokens)

    # Step 1: Retrieve context from uploaded document
    ctx_list = retrieve(question, k=k)

    # Step 2: Build system prompt including retrieved context
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

    # Step 3: Convert Streamlit history to LLM message format
    msgs = [{"role": "system", "content": system_prompt}]
    for h in history:
        msgs.append({"role": h["role"], "content": h["content"]})

    # Step 4: Append current question
    msgs.append({"role": "user", "content": question})

    # Step 5: Call Databricks LLM endpoint
    reply = call_llm(msgs, max_tokens=max_tokens)
    return reply, ctx_list

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Databricks Agentic RAG", page_icon="🧠")
st.title("🧠 Databricks Agentic RAG Chat")

client = get_deploy_client("databricks")
resp = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": ["test"]})
# st.write(resp)

# ensure resources at startup
ensure_table_and_index()

st.sidebar.header("Upload PDF")
f = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])
if st.sidebar.button("Ingest", disabled=not f):
    r = ingest_pdf(f.read(), f.name)
    st.sidebar.success(f"Ingested {r['chunks']} chunks (doc_id={r['doc_id']}) and index synced ✅")

if "history" not in st.session_state: st.session_state.history = []
for msg in st.session_state.history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

q = st.chat_input("Ask me anything…")
if q:
    st.session_state.history.append({"role":"user","content":q})
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking (Agentic)" if AGENTIC_MODE else "🤔 Thinking…"):
            ans, ctx = chat_with_rag(q, st.session_state.history[:-1])
        st.markdown(ans)
        with st.expander("Sources"):
            for c in ctx: st.markdown(f"📄 **{c['pdf_name']}** (p.{c['page']})\n\n> {c['content'][:300]}...")
    st.session_state.history.append({"role":"assistant","content":ans})