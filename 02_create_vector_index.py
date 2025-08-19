import os
from dotenv import load_dotenv
from databricks.vector_search.client import VectorSearchClient
load_dotenv()

VS_ENDPOINT, VS_INDEX, DELTA_TABLE = os.getenv("VS_ENDPOINT"), os.getenv("VS_INDEX"), os.getenv("DELTA_TABLE")
vsc = VectorSearchClient()
try: vsc.get_endpoint(VS_ENDPOINT)
except: vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
try: vsc.get_index(VS_ENDPOINT, VS_INDEX)
except:
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX,
        source_table_name=DELTA_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_dimension=1024,   # match your emb model
        embedding_vector_column="embedding",
        schema={"doc_id":"string","pdf_name":"string","page":"int","content":"string"},
    )
vsc.get_index(VS_ENDPOINT, VS_INDEX).sync()
print("Index ready")
