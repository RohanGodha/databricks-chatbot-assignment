from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

idx = w.vector_search_indexes.get_index("main.default.pdf_text_embeddings_index")
print(idx)

# from databricks.sdk import WorkspaceClient
# from databricks.sdk.errors.platform import InvalidParameterValue

# w = WorkspaceClient()

# ENDPOINT = "vs_endpoint_default"
# INDEX_NAME = "main.default.pdf_text_embeddings_index"


# def main():
#     print("🔎 Listing indexes under endpoint:", ENDPOINT)
#     indexes = w.vector_search_indexes.list_indexes(endpoint_name=ENDPOINT)

#     found = False
#     for idx in indexes:
#         found = True
#         print(f"- Name: {idx.name}, Type: {idx.index_type}")

#     if not found:
#         print("⚠️ No indexes found under this endpoint.")
#         return

#     print("\n📂 Scanning rows from index:", INDEX_NAME)
#     rows = w.vector_search_indexes.scan_index(
#         index_name=INDEX_NAME,
#         num_results=5
#     )

#     if not rows.data:
#         print("⚠️ No rows found in index (maybe ingestion didn’t work).")
#         return

#     for r in rows.data:
#         print("Row:", r)

#     print("\n💬 Running test query on index...")
#     try:
#         query = w.vector_search_indexes.query_index(
#             index_name=INDEX_NAME,
#             query_text="Summarize the document",
#             columns=["text"],
#             num_results=3
#         )
#         print("✅ Query executed successfully")

#         if not query.results:
#             print("⚠️ No results returned for query.")
#         else:
#             for res in query.results:
#                 print(res)

#     except InvalidParameterValue as e:
#         print("❌ Query failed:", e)
#         print("👉 This usually means the index is a **direct access index**.")
#         print("   In that case you must pass an embedding vector instead of query_text.")
#         print("   Example: use your embedding model to create a vector, then query with `query_vector=[...]`.")


# if __name__ == "__main__":
#     main()
