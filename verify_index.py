from databricks.vector_search.client import VectorSearchClient
from utils.embedder import embed_batch   # make sure you have this utility or write your own

# Initialize client
vsc = VectorSearchClient(disable_notice=True)

VS_ENDPOINT = "vs_endpoint_default"
VS_INDEX = "main.default.pdf_text_managed_vs_index"

# --- 1. Check Index Status ---
status = vsc.get_index(VS_ENDPOINT, VS_INDEX)
print("Index status:", status.describe())   # safer than printing object directly

# --- 2. Embed query text ---
query = "What is mentioned about financial risk?"

# embed_batch should return a list of embeddings (lists of floats)
q_emb = embed_batch([query])[0]  

# --- 3. Run similarity search with vector ---
results = vsc.get_index(VS_ENDPOINT, VS_INDEX).similarity_search(
    query_vector=q_emb,
    num_results=5,
    columns=["pdf_name", "content"]
)

# --- 4. Print results ---
for row in results.get("result", {}).get("data_array", []):
    print(f"File: {row[0]}\nContent: {row[1][:300]}...\n")
