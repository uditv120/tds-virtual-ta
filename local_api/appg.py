from __future__ import annotations
import os, json, traceback, re
from typing import List, Optional

import numpy as np
import requests, typesense
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# ENV CONFIG
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY", "xyz")
TYPESENSE_HOST = os.getenv("TYPESENSE_HOST", "localhost")
TYPESENSE_PORT = int(os.getenv("TYPESENSE_PORT", "8108"))
TYPESENSE_PROTOCOL = os.getenv("TYPESENSE_PROTOCOL", "http")

NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
NOMIC_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
NOMIC_MODEL = "nomic-embed-text-v1.5"

GITHUB_TOKEN = os.getenv("GIT_API_KEY")
GITHUB_ENDPOINT = "https://models.github.ai/inference"
GITHUB_MODEL = "openai/gpt-4.1"

COLLECTION_NAME = "tds_chunks"
TOP_K = 5
EMBED_DIM = 768

# FASTAPI INIT
app = FastAPI(title="Hybrid RAG + GitHub AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# TYPESENSE INIT
client = typesense.Client({
    "nodes": [{"host": TYPESENSE_HOST, "port": TYPESENSE_PORT, "protocol": TYPESENSE_PROTOCOL}],
    "api_key": TYPESENSE_API_KEY,
    "connection_timeout_seconds": 3,
})

# GITHUB AI CLIENT
ai_client = ChatCompletionsClient(
    endpoint=GITHUB_ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

class QueryRequest(BaseModel):
    question: str

# --- EMBEDDING ---
def embed_with_nomic(text: str) -> np.ndarray:
    headers = {
        "Authorization": f"Bearer {NOMIC_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": NOMIC_MODEL, "texts": [text]}
    response = requests.post(NOMIC_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    embedding = data.get("data", [{}])[0].get("embedding") or data.get("embeddings", [])[0]
    return np.array(embedding, dtype=np.float32)

# --- VECTOR SEARCH ---
def search_typesense_vector(query_vector, top_k=TOP_K):
    vector_str = ",".join(f"{v:.6f}" for v in query_vector.tolist())
    search_parameters = {
        "searches": [
            {
                "collection": COLLECTION_NAME,
                "q": "*",
                "query_by": "text",
                "vector_query": f"embedding:([{vector_str}], k:{top_k})"
            }
        ]
    }
    response = client.multi_search.perform(search_parameters)
    return response["results"][0]["hits"]

# --- PROMPT BUILDER ---
def build_prompt(user_q: str, chunks: List[dict]) -> str:
    context_texts = []
    for chunk in chunks:
        doc = chunk["document"]
        url = doc.get("url", "unknown")
        text = doc.get("text", "")[:160].replace("\n", " ").strip()
        context_texts.append(f"URL: {url}\nText: {text}")
    context = "\n\n---\n\n".join(context_texts)

    sample_response = {
        "answer": "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question.",
        "links": [
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
                "text": "Use the model that’s mentioned in the question."
            },
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3",
                "text": "My understanding is that you just have to use a tokenizer, similar to what Prof. Anand used, to get the number of tokens and multiply that by the given rate."
            }
        ]
    }

    return f"""
You are a helpful educational assistant for data-science learners.

Use ONLY the context below to answer the question. Provide a JSON response with an "answer" field and a "links" array containing exactly 2 helpful references (url + short summary).

Context:
{context}

Question:
{user_q}

Respond ONLY with a JSON object EXACTLY in this format:
{json.dumps(sample_response, indent=2)}

Do NOT add markdown, explanations, or extra formatting. Output only the JSON object.
""".strip()

# --- JSON EXTRACTOR ---
def parse_llm_response(raw_text: str):
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw_text)
    except:
        return {"answer": raw_text.strip(), "links": []}

# --- MAIN API ---
@app.post("/api/")
async def rag_answer(req: QueryRequest):
    try:
        question = req.question.strip()
        embedding = embed_with_nomic(question)
        chunks = search_typesense_vector(embedding, top_k=TOP_K)
        prompt = build_prompt(question, chunks)

        response = ai_client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=prompt),
            ],
            model=GITHUB_MODEL,
            temperature=0.3,
            top_p=1,
        )

        raw_text = response.choices[0].message.content
        return JSONResponse(content=parse_llm_response(raw_text))

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
