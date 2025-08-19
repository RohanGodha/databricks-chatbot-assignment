import os
from dotenv import load_dotenv
from databricks.vector_search.client import VectorSearchClient
from mlflow.deployments import get_deploy_client

load_dotenv()
VS_ENDPOINT, VS_INDEX = os.getenv("VS_ENDPOINT"), os.getenv("VS_INDEX")
CHAT_ENDPOINT = os.getenv("CHAT_ENDPOINT")
AGENTIC_MODE = os.getenv("AGENTIC_MODE","False").lower() == "true"

client = get_deploy_client("databricks")

def retrieve(query, k=5):
    vsc = VectorSearchClient()
    idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)
    res = idx.similarity_search(query_text=query, num_results=k, columns=["doc_id","pdf_name","page","content"])
    rows = res.get("result",{}).get("data_array",[])
    return [{"doc_id":r[0],"pdf_name":r[1],"page":r[2],"content":r[3]} for r in rows]

def call_llm(messages, max_tokens=400):
    resp = client.predict(endpoint=CHAT_ENDPOINT, inputs={"messages": messages, "max_tokens": max_tokens})
    if "choices" in resp: return resp["choices"][0]["message"]["content"]
    if "content" in resp: return resp["content"]
    return str(resp)[:2000]

def agentic_chat(question, history=None, max_loops=3, k=5):
    """Agent loop: ask LLM, let it decide whether to re-search or answer."""
    history = history or []
    for loop in range(max_loops):
        ctx = retrieve(question, k=k)
        system = (
            "You are an agent with access to the following tools:\n"
            " - Search(query): retrieve relevant text from documents.\n"
            "Instructions:\n"
            "1. Use ONLY provided context to answer.\n"
            "2. If context is insufficient, say: 'TOOL:SEARCH <better query>'.\n"
            "3. Otherwise, provide a final helpful answer with citations.\n\n"
            "Context:\n" + "\n---\n".join(c['content'] for c in ctx)
        )
        msgs = [{"role":"system","content":system}] + history + [{"role":"user","content":question}]
        reply = call_llm(msgs)

        # If model asks to re-search
        if reply.strip().upper().startswith("TOOL:SEARCH"):
            new_q = reply.split(" ",1)[1] if " " in reply else question
            question = new_q.strip()
            continue

        # Otherwise: final answer
        return reply, ctx
    return "I could not find enough context after multiple searches.", ctx

def chat_with_rag(question, history=None, **kwargs):
    if AGENTIC_MODE:
        return agentic_chat(question, history, **kwargs)
    else:
        # baseline: one-shot retrieval + answer
        ctx = retrieve(question, k=5)
        system = "Answer using ONLY this context. If missing, say 'I don't know'.\n\n" + "\n---\n".join(c["content"] for c in ctx)
        msgs = [{"role":"system","content":system}] + (history or []) + [{"role":"user","content":question}]
        reply = call_llm(msgs)
        return reply, ctx


# import os
# from dotenv import load_dotenv
# from databricks.vector_search.client import VectorSearchClient
# from mlflow.deployments import get_deploy_client

# load_dotenv()
# VS_ENDPOINT, VS_INDEX, CHAT_ENDPOINT = os.getenv("VS_ENDPOINT"), os.getenv("VS_INDEX"), os.getenv("CHAT_ENDPOINT")

# def retrieve(query, k=5):
#     vsc = VectorSearchClient()
#     idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)
#     res = idx.similarity_search(query_text=query, num_results=k, columns=["doc_id","pdf_name","page","content"])
#     rows = res.get("result",{}).get("data_array",[])
#     return [{"doc_id":r[0],"pdf_name":r[1],"page":r[2],"content":r[3]} for r in rows]

# def chat_with_rag(question, history=None, k=5, max_tokens=400):
#     history = history or []
#     ctx = retrieve(question, k=k)
#     system = "Answer using ONLY this context. If missing, say 'I don't know'.\n\n" + "\n---\n".join(c["content"] for c in ctx)
#     msgs = [{"role":"system","content":system}] + history + [{"role":"user","content":question}]
#     client = get_deploy_client("databricks")
#     resp = client.predict(endpoint=CHAT_ENDPOINT, inputs={"messages": msgs, "max_tokens": max_tokens})
#     reply = resp["choices"][0]["message"]["content"] if "choices" in resp else resp.get("content", str(resp))
#     return reply, ctx
