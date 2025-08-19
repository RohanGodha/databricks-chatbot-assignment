import logging
import os
import streamlit as st
import pdfplumber
from model_serving_utils import query_endpoint, is_endpoint_supported
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from databricks.vector_search.client import VectorSearchClient
# Read from environment variables (Streamlit secrets)

app = FastAPI()

DBX_HOST = os.getenv("DATABRICKS_HOST")
DBX_TOKEN = os.getenv("DATABRICKS_TOKEN")
SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT")
CHAT_ENDPOINT = os.getenv("CHAT_ENDPOINT")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")

# Initialize Databricks WorkspaceClient
w = WorkspaceClient(host=DBX_HOST, token=DBX_TOKEN)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
SERVING_ENDPOINT = "databricks-llama-4-maverick"
# os.getenv('SERVING_ENDPOINT')
assert SERVING_ENDPOINT, (
    "Unable to determine serving endpoint to use for chatbot app. If developing locally, "
    "set the SERVING_ENDPOINT environment variable to the name of your serving endpoint. If "
    "deploying to a Databricks app, include a serving endpoint resource named "
    "'serving_endpoint' with CAN_QUERY permissions."
)

# Check if endpoint is supported
endpoint_supported = is_endpoint_supported(SERVING_ENDPOINT)

def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

user_info = get_user_info()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant in a Databricks app."}
    ]
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = ""

# App title
st.title("🧱 Conversational AI Chatbot with PDF Support")

# PDF uploader
uploaded_file = st.file_uploader("📄 Upload a PDF for context", type=["pdf"])
if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        pdf_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    st.session_state.pdf_content = pdf_text.strip()
    st.success("✅ PDF uploaded and processed successfully!")

# Check endpoint compatibility
if not endpoint_supported:
    st.error("⚠️ Unsupported Endpoint Type")
    st.markdown(
        f"The endpoint `{SERVING_ENDPOINT}` is not compatible with this chatbot."
    )
else:
    st.markdown("💬 Chat below. Your history will persist for the session.")

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] != "system":  # don't display system prompt
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Type your message"):
        # Add user input to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare context if PDF uploaded
        messages_to_send = st.session_state.messages.copy()
        if st.session_state.pdf_content:
            messages_to_send.insert(1, {
                "role": "system",
                "content": f"The user has uploaded a PDF. Here is its content:\n{st.session_state.pdf_content}"
            })

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                assistant_reply = query_endpoint(
                    endpoint_name=SERVING_ENDPOINT,
                    messages=messages_to_send,
                    max_tokens=400,
                )["content"]
                st.markdown(assistant_reply)

        # Save assistant reply to history
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

vsc = VectorSearchClient(disable_notice=True)
VS_ENDPOINT = "vs_endpoint_default"
VS_INDEX = "main.default.pdf_text_managed_vs_index"

@app.post("/search")
def search_docs(query: str):
    results = vsc.get_index(VS_ENDPOINT, VS_INDEX).similarity_search(
        query_text=query,
        num_results=5,
        columns=["pdf_name", "content"]
    )
    return results