# inspect_delta_table.py

import os
from databricks import sql
from dotenv import load_dotenv

# --------------------
# Load environment variables
# --------------------
load_dotenv()
DB_HOST = os.getenv("DATABRICKS_HOST", "").replace("https://", "")
DB_HTTP_PATH = os.getenv("DATABRICKS_SQL_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DELTA_TABLE = os.getenv("DELTA_TABLE")

# def main():
#     # Connect to Databricks SQL
#     conn = sql.connect(
#         server_hostname=DB_HOST,
#         http_path=DB_HTTP_PATH,
#         access_token=DATABRICKS_TOKEN
#     )
    
#     with conn.cursor() as cursor:
#         # 1️⃣ Get all PDF names
#         cursor.execute(f"SELECT DISTINCT pdf_name FROM {DELTA_TABLE}")
#         pdfs = [row[0] for row in cursor.fetchall()]
#         if not pdfs:
#             print("No PDFs found in the table.")
#             return

#         for pdf in pdfs:
#             print(f"\n===== Top 5 chunks for PDF: {pdf} =====")
#             cursor.execute(
#                 f"""
#                 SELECT doc_id, page, chunk_id, content, created_at 
#                 FROM {DELTA_TABLE} 
#                 WHERE pdf_name = ? 
#                 ORDER BY created_at DESC 
#                 LIMIT 5
#                 """,
#                 (pdf,)
#             )
#             rows = cursor.fetchall()
#             if not rows:
#                 print("No chunks found for this PDF.")
#                 continue
#             for r in rows:
#                 doc_id, page, chunk_id, content, created_at = r
#                 print(f"\nPage: {page}, Chunk ID: {chunk_id}, Doc ID: {doc_id}, Created: {created_at}")
#                 print(f"Content: {content[:300]}{'...' if len(content) > 300 else ''}")
    
#     conn.close()
#     print("\n✅ Done printing top 5 chunks per PDF.")

# if __name__ == "__main__":
#     main()

# --------------------
# Connect to Databricks SQL
# --------------------
conn = sql.connect(
    server_hostname=DB_HOST,
    http_path=DB_HTTP_PATH,
    access_token=DATABRICKS_TOKEN
)

try:
    with conn.cursor() as cursor:
        # 1️⃣ List distinct PDFs in the table
        cursor.execute(f"SELECT DISTINCT pdf_name FROM {DELTA_TABLE}")
        pdfs = [r[0] for r in cursor.fetchall()]
        if not pdfs:
            print("No PDFs found in the table.")
        else:
            print(f"Found PDFs: {pdfs}\n")

        # 2️⃣ For each PDF, print top 5 rows and check embeddings
        for pdf in pdfs:
            print(f"================ PDF: {pdf} ================\n")
            
            # Top 5 rows
            cursor.execute(
                f"SELECT doc_id, page, chunk_id, content, embedding FROM {DELTA_TABLE} "
                f"WHERE pdf_name = ? LIMIT 5",
                (pdf,)
            )
            rows = cursor.fetchall()
            print(f"Top 5 chunks for {pdf}:")
            for r in rows:
                doc_id, page, chunk_id, content, embedding = r
                emb_status = "✅" if embedding is not None and len(embedding) > 0 else "⚠️ MISSING"
                print(f"Page {page}, Chunk ID {chunk_id}, Doc ID {doc_id}, Embedding: {emb_status}")
                print(f"Content preview: {content[:100]}...\n")

            # Count total chunks
            cursor.execute(
                f"SELECT COUNT(*) FROM {DELTA_TABLE} WHERE pdf_name = ?",
                (pdf,)
            )
            total_chunks = cursor.fetchone()[0]

            # Count chunks with embeddings
            cursor.execute(
                f"SELECT COUNT(*) FROM {DELTA_TABLE} WHERE pdf_name = ? AND embedding IS NOT NULL AND size(embedding) > 0",
                (pdf,)
            )
            emb_chunks = cursor.fetchone()[0]

            # Chunks missing embeddings
            cursor.execute(
                f"SELECT COUNT(*) FROM {DELTA_TABLE} WHERE pdf_name = ? AND (embedding IS NULL OR size(embedding) = 0)",
                (pdf,)
            )
            missing_emb = cursor.fetchone()[0]

            print(f"Total chunks: {total_chunks}")
            print(f"Chunks with embeddings: {emb_chunks}")
            print(f"Chunks missing embeddings: {missing_emb}\n")

finally:
    conn.close()