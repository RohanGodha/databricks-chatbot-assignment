from databricks import sql

conn = sql.connect(
    server_hostname="dbc-3cf3bb0b-20e2.cloud.databricks.com",
    http_path="/sql/1.0/warehouses/46fd3aa8d2dd5dba",
    access_token="dapi60cbb451dffd5c7e6217facb700ef401"
)

cursor = conn.cursor()

cursor.execute("SHOW SCHEMAS")

cursor.execute("""
CREATE TABLE IF NOT EXISTS main.default.pdf_text_embeddings (
  pk STRING,
  pdf_name STRING,
  content STRING,
  embedding ARRAY<FLOAT>
) USING DELTA
""")

print("Table created successfully in Unity Catalog")

cursor.close()
conn.close()