from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

ENDPOINT = "vs_endpoint_default"
INDEX_NAME = "main.default.pdf_text_embeddings_index"

def main():
    print("🔎 Listing indexes under endpoint:", ENDPOINT)
    indexes = w.vector_search_indexes.list_indexes(endpoint_name=ENDPOINT)

    found = False
    for idx in indexes:
        found = True
        print(f"- Name: {idx.name}, Type: {idx.index_type}")

    if not found:
        print("⚠️ No indexes found under this endpoint.")

    print("\n📂 Scanning rows from index:", INDEX_NAME)
    rows = w.vector_search_indexes.scan_index(
        index_name=INDEX_NAME,
        max_results=5  # ✅ removed 'endpoint_name'
    )

    if not rows.data:
        print("⚠️ No rows found in index (maybe ingestion didn’t work).")
    else:
        for r in rows.data:
            print(r)

    print("\n💬 Running test query on index...")
    query = w.vector_search_indexes.query_index(
        index_name=INDEX_NAME,
        query_text="Summarize the document",
        num_results=3  # ✅ removed 'endpoint_name'
    )

    if not query.results:
        print("⚠️ No results returned for query.")
    else:
        for res in query.results:
            print(res)

if __name__ == "__main__":
    main()
