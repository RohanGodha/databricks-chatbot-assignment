import logging
import os
import streamlit as st
import pdfplumber
from model_serving_utils import query_endpoint, is_endpoint_supported
from databricks.sdk import WorkspaceClient

# Read from environment variables (Streamlit secrets)
DBX_HOST = "https://dbc-3cf3bb0b-20e2.cloud.databricks.com"
DBX_TOKEN = "dapi60cbb451dffd5c7e6217facb700ef401"

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


# import logging
# import os
# import streamlit as st
# from model_serving_utils import query_endpoint, is_endpoint_supported

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Ensure environment variable is set correctly
# SERVING_ENDPOINT = os.getenv('SERVING_ENDPOINT')
# assert SERVING_ENDPOINT, (
#     "Unable to determine serving endpoint to use for chatbot app. If developing locally, "
#     "set the SERVING_ENDPOINT environment variable to the name of your serving endpoint. If "
#     "deploying to a Databricks app, include a serving endpoint resource named "
#     "'serving_endpoint' with CAN_QUERY permissions."
# )

# # Check if endpoint is supported
# endpoint_supported = is_endpoint_supported(SERVING_ENDPOINT)

# def get_user_info():
#     headers = st.context.headers
#     return dict(
#         user_name=headers.get("X-Forwarded-Preferred-Username"),
#         user_email=headers.get("X-Forwarded-Email"),
#         user_id=headers.get("X-Forwarded-User"),
#     )

# user_info = get_user_info()

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "system", "content": "You are a helpful assistant in a Databricks app."}
#     ]

# # App title
# st.title("🧱 Conversational AI Chatbot")

# # Check endpoint compatibility
# if not endpoint_supported:
#     st.error("⚠️ Unsupported Endpoint Type")
#     st.markdown(
#         f"The endpoint `{SERVING_ENDPOINT}` is not compatible with this chatbot."
#     )
# else:
#     st.markdown("💬 Chat below. Your history will persist for the session.")

#     # Display chat history
#     for message in st.session_state.messages:
#         if message["role"] != "system":  # don't display system prompt
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#     # User input
#     if prompt := st.chat_input("Type your message"):
#         # Add user input to history
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Assistant response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 assistant_reply = query_endpoint(
#                     endpoint_name=SERVING_ENDPOINT,
#                     messages=st.session_state.messages,
#                     max_tokens=400,
#                 )["content"]
#                 st.markdown(assistant_reply)

#         # Save assistant reply to history
#         st.session_state.messages.append({"role": "assistant", "content": assistant_reply})


# # import logging
# # import os
# # import streamlit as st
# # from model_serving_utils import query_endpoint, is_endpoint_supported

# # # Set up logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # Ensure environment variable is set correctly
# # SERVING_ENDPOINT = os.getenv('SERVING_ENDPOINT')
# # assert SERVING_ENDPOINT, \
# #     ("Unable to determine serving endpoint to use for chatbot app. If developing locally, "
# #      "set the SERVING_ENDPOINT environment variable to the name of your serving endpoint. If "
# #      "deploying to a Databricks app, include a serving endpoint resource named "
# #      "'serving_endpoint' with CAN_QUERY permissions, as described in "
# #      "https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app#deploy-the-databricks-app")

# # # Check if the endpoint is supported
# # endpoint_supported = is_endpoint_supported(SERVING_ENDPOINT)

# # def get_user_info():
# #     headers = st.context.headers
# #     return dict(
# #         user_name=headers.get("X-Forwarded-Preferred-Username"),
# #         user_email=headers.get("X-Forwarded-Email"),
# #         user_id=headers.get("X-Forwarded-User"),
# #     )

# # user_info = get_user_info()

# # # Streamlit app
# # if "visibility" not in st.session_state:
# #     st.session_state.visibility = "visible"
# #     st.session_state.disabled = False

# # st.title("🧱 Chatbot App")

# # # Check if endpoint is supported and show appropriate UI
# # if not endpoint_supported:
# #     st.error("⚠️ Unsupported Endpoint Type")
# #     st.markdown(
# #         f"The endpoint `{SERVING_ENDPOINT}` is not compatible with this basic chatbot template.\n\n"
# #         "This template only supports chat completions-compatible endpoints.\n\n"
# #         "👉 **For a richer chatbot template** that supports all conversational endpoints on Databricks, "
# #         "please see the [Databricks documentation](https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app)."
# #     )
# # else:
# #     st.markdown(
# #         "ℹ️ This is a simple example. See "
# #         "[Databricks docs](https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app) "
# #         "for a more comprehensive example with streaming output and more."
# #     )

# #     # Initialize chat history
# #     if "messages" not in st.session_state:
# #         st.session_state.messages = []

# #     # Display chat messages from history on app rerun
# #     for message in st.session_state.messages:
# #         with st.chat_message(message["role"]):
# #             st.markdown(message["content"])

# #     # Accept user input
# #     if prompt := st.chat_input("What is up?"):
# #         # Add user message to chat history
# #         st.session_state.messages.append({"role": "user", "content": prompt})
# #         # Display user message in chat message container
# #         with st.chat_message("user"):
# #             st.markdown(prompt)

# #         # Display assistant response in chat message container
# #         with st.chat_message("assistant"):
# #             # Query the Databricks serving endpoint
# #             assistant_response = query_endpoint(
# #                 endpoint_name=SERVING_ENDPOINT,
# #                 messages=st.session_state.messages,
# #                 max_tokens=400,
# #             )["content"]
# #             st.markdown(assistant_response)


# #         # Add assistant response to chat history
# #         st.session_state.messages.append({"role": "assistant", "content": assistant_response})
