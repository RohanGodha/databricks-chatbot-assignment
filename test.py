from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()
print(vsc.list_endpoints())