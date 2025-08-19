from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

VS_ENDPOINT = "vs_endpoint_default"
VS_INDEX = "main.default.pdf_text_embeddings_index"

# Delete the index
try:
    vsc.delete_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
    print(f"Deleted index {VS_INDEX} ✅")
except Exception as e:
    print(f"Error deleting index: {e}")