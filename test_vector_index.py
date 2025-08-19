from databricks.vector_search.client import VectorSearchClient
import os

# Load from environment (make sure these are set)
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    raise ValueError("Please set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables")

# Initialize client
vsc = VectorSearchClient(
    workspace_url=DATABRICKS_HOST,
    personal_access_token=DATABRICKS_TOKEN
)

print("[NOTICE] Using Vector Search test script")

# Target database and table
database_name = "main"
source_table = "default.pdf_text"
index_name = "pdf_text_managed_vs_index"
endpoint_name = "vs_demo_endpoint"

try:
    print(f"Creating endpoint {endpoint_name} (if not exists)...")
    vsc.create_endpoint_and_wait(
        name=endpoint_name,
        endpoint_type="STANDARD"
    )
    print("✅ Endpoint ready")
except Exception as e:
    print(f"⚠️ Endpoint may already exist: {e}")

try:
    print(f"Creating managed index {database_name}.{index_name}...")

    resp = vsc.create_delta_sync_index_and_wait(
        index_name=f"{database_name}.{index_name}",
        endpoint_name=endpoint_name,
        source_table_name=source_table,
        primary_key="id",  # Make sure `id` exists in your table
        embedding_dimension=768,  # match your embeddings size
        pipeline_type="TRIGGERED"  # <-- REQUIRED (or use "CONTINUOUS")
    )

    print("✅ Index created successfully:")
    print(resp)

except Exception as e:
    print(f"❌ Error creating index: {e}")

print("Done.")


# from databricks.vector_search.client import VectorSearchClient

# print("[NOTICE] Using Vector Search test script")

# # Initialize client
# vsc = VectorSearchClient(
#     workspace_url="https://<your-databricks-workspace-url>",
#     personal_access_token="<your-pat>"
# )

# # Index details
# endpoint_name = "vector-search-endpoint"
# index_name = "main.default.pdf_text_managed_vs_index"
# source_table = "main.default.pdf_text_embeddings"  # must already exist

# # Try deleting old index
# print(f"Trying to delete old index {index_name} (if exists)...")
# try:
#     vsc.delete_index(endpoint_name, index_name)
#     print(f"✅ Deleted old index {index_name}")
# except Exception as e:
#     print(f"⚠️ Could not delete old index (maybe not found): {e}")

# # Create new managed index
# print(f"Creating managed index {index_name}...")

# try:
#     result = vsc.create_delta_sync_index_and_wait(
#         endpoint_name=endpoint_name,
#         index_name=index_name,
#         source_table_name=source_table,
#         primary_key="id",  # make sure your table has this column
#         embedding_vector_column="embedding"  # column with embeddings
#     )
#     print(f"✅ Successfully created index: {result}")
# except Exception as e:
#     print(f"❌ Error creating index: {e}")

# print("Done.")


# # from databricks.vector_search.client import VectorSearchClient
# # import os

# # # Initialize client (uses default notebook or PAT authentication)
# # vsc = VectorSearchClient(disable_notice=True)
# # # Print all available attributes and methods
# # print("Available attributes/methods in VectorSearchClient:")
# # for attr in dir(vsc):
# #     print(attr)


# # INDEX_NAME = "main.default.pdf_text_managed_vs_index"
# # ENDPOINT_NAME = "vs_endpoint_default"

# # print("[NOTICE] Using Vector Search test script")

# # # 1. Try deleting old index
# # print(f"Trying to delete old index {INDEX_NAME} (if exists)...")
# # try:
# #     vsc.vector_indexes.delete_index(
# #         endpoint_name=ENDPOINT_NAME,
# #         index_name=INDEX_NAME
# #     )
# #     print(f"✅ Deleted old index {INDEX_NAME}")
# # except Exception as e:
# #     print(f"No old index found or error deleting: {e}")

# # # 2. Try creating a new managed index
# # print(f"Creating managed index {INDEX_NAME}...")
# # try:
# #     vsc.vector_indexes.create_index(
# #         endpoint_name=ENDPOINT_NAME,
# #         index_name=INDEX_NAME,
# #         primary_key="id",
# #         index_type="managed",
# #         schema={
# #             "id": "string",
# #             "text": "string",
# #             "vector": "array<float>"
# #         },
# #         embedding_source_column="text",
# #         embedding_model_endpoint_name="databricks-bge-large-en"
# #     )
# #     print(f"✅ Created managed index {INDEX_NAME}")
# # except Exception as e:
# #     print(f"❌ Error creating index: {e}")

# # print("Done.")



# # # from databricks.vector_search.client import VectorSearchClient

# # # print("[NOTICE] Using Vector Search test script")

# # # # Initialize client
# # # client = VectorSearchClient()

# # # # Endpoint & index names
# # # endpoint_name = "vs_endpoint_default"
# # # index_name = "main.default.pdf_text_managed_vs_index"

# # # # 1. Try to delete the old index
# # # print(f"Trying to delete old index {index_name} (if exists)...")
# # # try:
# # #     client.delete_index(endpoint_name=endpoint_name, index_name=index_name)
# # #     print(f"✅ Deleted old index {index_name}")
# # # except Exception as e:
# # #     print(f"No old index found or error deleting: {e}")

# # # # 2. Create the index
# # # print(f"Creating managed index {index_name}...")
# # # try:
# # #     client.create_index(
# # #         endpoint_name=endpoint_name,
# # #         index_name=index_name,
# # #         index_type="DELTA_TABLE",
# # #         delta_table_name="main.default.pdf_text",  # must exist!
# # #         primary_key="id"  # column in your Delta table
# # #     )
# # #     print(f"✅ Index {index_name} created successfully")
# # # except Exception as e:
# # #     print(f"❌ Error creating index: {e}")

# # # print("Done.")


