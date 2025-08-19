from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

VS_ENDPOINT = "vs_endpoint_default"
VS_INDEX = "main.default.pdf_text_managed_vs_index"
EMBED_TABLE = "main.default.pdf_text_embeddings"

# ---- Step 1: Endpoint ----
eps = {e["name"] for e in vsc.list_endpoints().get("endpoints", [])}
if VS_ENDPOINT not in eps:
    print(f"Creating endpoint: {VS_ENDPOINT}")
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
else:
    print(f"Endpoint already exists: {VS_ENDPOINT}")

# ---- Step 2: Index ----
indexes = {i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])}
if VS_INDEX not in indexes:
    print(f"Creating index: {VS_INDEX}")
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX,
        source_table_name=EMBED_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="pk",
        embedding_dimension=1024,
        embedding_vector_column="embedding"
    )
else:
    print(f"Index already exists: {VS_INDEX}")


# from databricks.vector_search.client import VectorSearchClient
# from databricks import sql

# # Initialize Vector Search client
# vsc = VectorSearchClient(disable_notice=True)

# VS_ENDPOINT = "vs_endpoint_default"
# VS_INDEX = "main.default.pdf_text_managed_vs_index"
# EMBED_TABLE = "main.default.pdf_text_embeddings"

# # ---- Step 1: Enable Change Data Feed (CDF) on the source table ----
# # Run SQL against Databricks
# connection = sql.connect(
#     server_hostname="dbc-3cf3bb0b-20e2.cloud.databricks.com",  # replace with your workspace hostname
#     http_path="/sql/1.0/warehouses/<warehouse-id>",             # replace with your SQL warehouse path
#     access_token="<your-databricks-pat>"                        # replace with your PAT
# )

# cursor = connection.cursor()
# cursor.execute(f"""
#   ALTER TABLE {EMBED_TABLE}
#   SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
# """)
# cursor.close()
# connection.close()

# print(f"✅ Change Data Feed enabled for table: {EMBED_TABLE}")

# # ---- Step 2: Ensure endpoint exists ----
# eps = {e["name"] for e in vsc.list_endpoints().get("endpoints", [])}
# if VS_ENDPOINT not in eps:
#     print(f"Creating endpoint: {VS_ENDPOINT}")
#     vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
# else:
#     print(f"Endpoint already exists: {VS_ENDPOINT}")

# # ---- Step 3: Ensure index exists ----
# indexes = {i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])}
# if VS_INDEX not in indexes:
#     print(f"Creating index: {VS_INDEX}")
#     vsc.create_delta_sync_index(
#         endpoint_name=VS_ENDPOINT,
#         index_name=VS_INDEX,
#         source_table_name=EMBED_TABLE,
#         pipeline_type="TRIGGERED",   # index sync type
#         primary_key="pk",
#         embedding_dimension=1024,
#         embedding_vector_column="embedding"
#     )
# else:
#     print(f"Index already exists: {VS_INDEX}")


# # from databricks.vector_search.client import VectorSearchClient

# # vsc = VectorSearchClient(disable_notice=True)

# # VS_ENDPOINT = "vs_endpoint_default"
# # VS_INDEX = "main.default.pdf_text_managed_vs_index"
# # EMBED_TABLE = "main.default.pdf_text_embeddings"

# # eps = {e["name"] for e in vsc.list_endpoints().get("endpoints", [])}
# # if VS_ENDPOINT not in eps:
# #     print(f"Creating endpoint: {VS_ENDPOINT}")
# #     vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
# # else:
# #     print(f"Endpoint already exists: {VS_ENDPOINT}")

# # indexes = {i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])}
# # if VS_INDEX not in indexes:
# #     print(f"Creating index: {VS_INDEX}")
# #     vsc.create_delta_sync_index(
# #     endpoint_name=VS_ENDPOINT,
# #     index_name=VS_INDEX,
# #     source_table_name=EMBED_TABLE,
# #     pipeline_type="TRIGGERED",
# #     primary_key="pk",
# #     embedding_dimension=1024,
# #     embedding_vector_column="embedding"
# # )


# # else:
# #     print(f"Index already exists: {VS_INDEX}")
