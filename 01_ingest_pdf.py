import os, io, uuid, datetime as dt
import pdfplumber
from dotenv import load_dotenv
from databricks import sql
from mlflow.deployments import get_deploy_client

load_dotenv()
DELTA_TABLE, EMBEDDING_ENDPOINT = os.getenv("DELTA_TABLE"), os.getenv("EMBEDDING_ENDPOINT")

def chunk_text(text, max_chars=1200, overlap=200):
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        k = text.rfind(" ", i + max_chars - 200, j)
        j = k if k != -1 else j
        chunks.append(text[i:j])
        i = max(j - overlap, j)
    return [c for c in chunks if c.strip()]

def embed_chunks(chunks):
    client = get_deploy_client("databricks")
    out = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": chunks})
    return [row["embedding"] for row in out["data"]]

def write_rows(rows):
    host = os.getenv("DATABRICKS_HOST").replace("https://","")
    conn = sql.connect(server_hostname=host, http_path=os.getenv("DATABRICKS_SQL_HTTP_PATH"), access_token=os.getenv("DATABRICKS_TOKEN"))
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
    return {"doc_id": doc_id, "chunks": len(rows)}