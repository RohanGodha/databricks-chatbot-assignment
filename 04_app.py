import os, io, uuid, datetime as dt
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from databricks import sql
from databricks.vector_search.client import VectorSearchClient
from mlflow.deployments import get_deploy_client

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
DB_TOKEN = os.getenv("DATABRICKS_TOKEN")

client = get_deploy_client("databricks")
vsc = VectorSearchClient()

# --------------------
# Ensure Delta table + VS index
# --------------------
# def ensure_table_and_index(dim=1024):
#     conn = sql.connect(server_hostname=DB_HOST, http_path=DB_HTTP_PATH, access_token=DB_TOKEN)
#     with conn.cursor() as c:
#         c.execute(f"""CREATE TABLE IF NOT EXISTS {DELTA_TABLE} (
#             doc_id STRING, pdf_name STRING, page INT, chunk_id STRING,
#             content STRING, embedding ARRAY<FLOAT>, created_at TIMESTAMP
#         ) USING DELTA""")
#     conn.close()

#     try: vsc.get_endpoint(VS_ENDPOINT)
#     except: vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")

#     try: vsc.get_index(VS_ENDPOINT, VS_INDEX)
#     except:
#         vsc.create_delta_sync_index(
#             endpoint_name=VS_ENDPOINT,
#             index_name=VS_INDEX,
#             source_table_name=DELTA_TABLE,
#             pipeline_type="TRIGGERED",
#             primary_key="chunk_id",
#             embedding_dimension=dim,
#             embedding_vector_column="embedding",
#             schema={"doc_id":"string","pdf_name":"string","page":"int","content":"string"},
#         )

def ensure_table_and_index(dim=1024):
    # Ensure table
    conn = sql.connect(server_hostname=DB_HOST, http_path=DB_HTTP_PATH, access_token=DB_TOKEN)
    with conn.cursor() as c:
        c.execute(f"""CREATE TABLE IF NOT EXISTS {DELTA_TABLE} (
            doc_id STRING, pdf_name STRING, page INT, chunk_id STRING,
            content STRING, embedding ARRAY<FLOAT>, created_at TIMESTAMP
        ) USING DELTA""")
    conn.close()

    # Ensure endpoint (reuse existing if quota=1)
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

    # Ensure index
    try:
        vsc.get_index(VS_ENDPOINT, VS_INDEX)
    except Exception:
        vsc.create_delta_sync_index(
            endpoint_name=VS_ENDPOINT,
            index_name=VS_INDEX,
            source_table_name=DELTA_TABLE,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_dimension=dim,
            embedding_vector_column="embedding"
        )


def sync_index():
    vsc.get_index(VS_ENDPOINT, VS_INDEX).sync()

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
    out = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": chunks})
    return [row["embedding"] for row in out["data"]]

def write_rows(rows):
    conn = sql.connect(server_hostname=DB_HOST, http_path=DB_HTTP_PATH, access_token=DB_TOKEN)
    with conn.cursor() as c:
        c.executemany(f"INSERT INTO {DELTA_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.close()

def ingest_pdf(file_bytes, pdf_name):
    doc_id, rows = str(uuid.uuid4()), []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for pageno, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            chunks = chunk_text(text)
            if not chunks: continue
            embs = embed_chunks(chunks)
            now = dt.datetime.utcnow()
            for ch, emb in zip(chunks, embs):
                rows.append((doc_id, pdf_name, pageno, str(uuid.uuid4()), ch, emb, now))
    if rows: write_rows(rows)
    sync_index()  # auto sync index after ingestion
    return {"doc_id": doc_id, "chunks": len(rows)}

# --------------------
# Agentic Retriever
# --------------------
def retrieve(query, k=5):
    idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)
    res = idx.similarity_search(query_text=query, num_results=k, columns=["doc_id","pdf_name","page","content"])
    rows = res.get("result",{}).get("data_array",[])
    return [{"doc_id":r[0],"pdf_name":r[1],"page":r[2],"content":r[3]} for r in rows]

def call_llm(messages, max_tokens=400):
    resp = client.predict(endpoint=CHAT_ENDPOINT, inputs={"messages": messages, "max_tokens": max_tokens})
    if "choices" in resp: return resp["choices"][0]["message"]["content"]
    if "content" in resp: return resp["content"]
    return str(resp)[:2000]

def agentic_chat(question, history=None, max_loops=3, k=5):
    history = history or []
    for _ in range(max_loops):
        ctx = retrieve(question, k=k)
        system = (
            "You are an agent with tool access.\n"
            "Tools:\n - Search(query): retrieves from documents.\n"
            "If insufficient context, reply with: TOOL:SEARCH <better query>\n"
            "Otherwise answer with citations.\n\n"
            "Context:\n" + "\n---\n".join(c['content'] for c in ctx)
        )
        msgs = [{"role":"system","content":system}] + history + [{"role":"user","content":question}]
        reply = call_llm(msgs)
        if reply.strip().upper().startswith("TOOL:SEARCH"):
            question = reply.split(" ",1)[1] if " " in reply else question
            continue
        return reply, ctx
    return "I could not find enough context.", ctx

def chat_with_rag(question, history=None, **kwargs):
    if AGENTIC_MODE: return agentic_chat(question, history, **kwargs)
    ctx = retrieve(question, k=5)
    system = "Answer using ONLY this context. If missing, say 'I don't know'.\n\n" + "\n---\n".join(c["content"] for c in ctx)
    msgs = [{"role":"system","content":system}] + (history or []) + [{"role":"user","content":question}]
    return call_llm(msgs), ctx

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Databricks Agentic RAG", page_icon="🧠")
st.title("🧠 Databricks Agentic RAG Chat")

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


# import streamlit as st
# from dotenv import load_dotenv
# from 01_ingest_pdf import ingest_pdf
# from 03_retriever import chat_with_rag, AGENTIC_MODE

# load_dotenv()
# st.set_page_config(page_title="Databricks RAG Agent", page_icon="🧠")
# st.title("🧠 Databricks Agentic RAG Chat")

# st.sidebar.header("Upload PDF")
# f = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])
# if st.sidebar.button("Ingest", disabled=not f):
#     r = ingest_pdf(f.read(), f.name)
#     st.sidebar.success(f"Ingested {r['chunks']} chunks (doc_id={r['doc_id']}). Remember to sync index!")

# if "history" not in st.session_state: st.session_state.history = []
# for msg in st.session_state.history:
#     with st.chat_message(msg["role"]): st.markdown(msg["content"])

# q = st.chat_input("Ask me anything about your documents…")
# if q:
#     st.session_state.history.append({"role":"user","content":q})
#     with st.chat_message("assistant"):
#         with st.spinner("🔍 Thinking (agent mode)" if AGENTIC_MODE else "🤔 Thinking…"):
#             ans, ctx = chat_with_rag(q, st.session_state.history[:-1])
#         st.markdown(ans)
#         with st.expander("Sources"):
#             for c in ctx: st.markdown(f"📄 **{c['pdf_name']}** (p.{c['page']})\n\n> {c['content'][:300]}...")
#     st.session_state.history.append({"role":"assistant","content":ans})
