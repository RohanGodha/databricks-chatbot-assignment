from mlflow.deployments import get_deploy_client
import os
from dotenv import load_dotenv

# --------------------
# Load environment variables
# --------------------
load_dotenv()
client = get_deploy_client("databricks")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")
print("EMBEDDING_ENDPOINT--->")
print(EMBEDDING_ENDPOINT)
# test with a dummy input
test_texts = ["Hello world", "Test embedding"]

try:
    resp = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": test_texts})
    print("✅ Embedding endpoint response keys:", resp.keys())
    for i, row in enumerate(resp.get("data", [])):
        print(f"Text {i}: embedding length={len(row.get('embedding', []))}")
except Exception as e:
    print("❌ Error calling embedding endpoint:", e)
