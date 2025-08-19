import os, sys
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from mlflow.deployments import get_deploy_client
from databricks import sql
from databricks.vector_search.client import VectorSearchClient

load_dotenv()
ok = lambda m: print(f"✅ {m}")
bad = lambda m: print(f"❌ {m}")

try:
    w = WorkspaceClient()
    me = w.current_user.me()
    ok(f"Workspace user: {me.user_name}")
except Exception as e:
    bad(f"Workspace: {e}"); sys.exit(1)

client = get_deploy_client("databricks")
chat_ep, emb_ep = os.getenv("CHAT_ENDPOINT"), os.getenv("EMBEDDING_ENDPOINT")
for ep in [chat_ep, emb_ep]:
    try:
        info = client.get_endpoint(ep)
        ok(f"Endpoint {ep} → {info.get('state',{}).get('ready',{}).get('value')}")
    except Exception as e:
        bad(f"Endpoint {ep}: {e}"); sys.exit(1)

# Probe embedding dim
try:
    out = client.predict(endpoint=emb_ep, inputs={"input": ["dim check"]})
    emb = out["data"][0]["embedding"] if "data" in out else out[0]["embedding"]
    DIM = len(emb)
    ok(f"Embedding dimension = {DIM}")
except Exception as e:
    bad(f"Embedding call: {e}"); sys.exit(1)

# Ensure table
host = os.getenv("DATABRICKS_HOST").replace("https://","")
http_path, token = os.getenv("DATABRICKS_SQL_HTTP_PATH"), os.getenv("DATABRICKS_TOKEN")
table = os.getenv("DELTA_TABLE")
try:
    conn = sql.connect(server_hostname=host, http_path=http_path, access_token=token)
    with conn.cursor() as c:
        c.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
            doc_id STRING, pdf_name STRING, page INT, chunk_id STRING,
            content STRING, embedding ARRAY<FLOAT>, created_at TIMESTAMP
        ) USING DELTA""")
    conn.close()
    ok(f"Table {table} ensured")
except Exception as e:
    bad(f"Table error: {e}"); sys.exit(1)

# Ensure VS
vs_ep, vs_index = os.getenv("VS_ENDPOINT"), os.getenv("VS_INDEX")
try:
    vsc = VectorSearchClient()
    try: vsc.get_endpoint(vs_ep)
    except: vsc.create_endpoint(name=vs_ep, endpoint_type="STANDARD")
    try: vsc.get_index(vs_ep, vs_index)
    except:
        vsc.create_delta_sync_index(
            endpoint_name=vs_ep,
            index_name=vs_index,
            source_table_name=table,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_dimension=DIM,
            embedding_vector_column="embedding",
            schema={"doc_id":"string","pdf_name":"string","page":"int","content":"string"},
        )
    vsc.get_index(vs_ep, vs_index).sync()
    ok(f"VS index {vs_index} ready & synced")
except Exception as e:
    bad(f"VS setup: {e}"); sys.exit(1)

ok("Validation complete")