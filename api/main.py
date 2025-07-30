from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import tempfile
import os
import json

from agno.agent import Agent
from agno.models.google import Gemini
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.pineconedb import PineconeDb
from agno.embedder.google import GeminiEmbedder

# === FastAPI App Initialization ===
app = FastAPI()

# ...existing code...

# === API Keys ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# === Pinecone Setup ===
PINECONE_INDEX_NAME = "hackrx-policy-index-v2"
vector_db = PineconeDb(
    name=PINECONE_INDEX_NAME,
    dimension=1536,
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    api_key=PINECONE_API_KEY,
    embedder=GeminiEmbedder(api_key=GOOGLE_API_KEY_2),
    use_hybrid_search=True,
    hybrid_alpha=0.5,
)

# === Pydantic Models ===
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# === Endpoint ===
@app.post("/hackrx/run", response_model=QueryResponse)
def ask_document_questions(
    body: QueryRequest,
    authorization: str = Header(...)
):
    # === Bearer Token Validation ===
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    # === Download PDF ===
    try:
        response = requests.get(body.documents, timeout=30)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            pdf_path = temp_pdf.name
        print(f"✅ PDF downloaded: {pdf_path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}")

    # === Load PDF ===
    try:
        knowledge_base = PDFKnowledgeBase(path=pdf_path, vector_db=vector_db)
        knowledge_base.load(recreate=True, upsert=True)
        print(f"✅ Knowledge base loaded.")

        agent = Agent(
            model=Gemini(id="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
            knowledge=knowledge_base,
            show_tool_calls=True,
            markdown=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB error: {str(e)}")

    # === Prompt Construction ===
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(body.questions)])
    prompt = f"""You are a helpful assistant.

    Based on the uploaded document, answer the following questions only using the document content.

    Return the answers in this exact format:
    {{ "answers": [ "Answer to question 1", "Answer to question 2", "..." ] }}

    Do not include triple backticks or markdown formatting. Just provide answer for the each question in a single line.

    Questions:
    {questions_text}
    """

    try:
        # Call Gemini
        response = agent.run(prompt)
        raw_output = response.content.strip()
        print(f"🧠 Raw Gemini Response:\n{raw_output}\n")

        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output.replace("```", "").strip()

        # Try parsing the raw output as JSON
        answers_json = json.loads(raw_output)
        answers = answers_json.get("answers", [])

        if not isinstance(answers, list):
            raise ValueError("Invalid JSON structure: 'answers' is not a list.")

    except Exception as e:
        print(f"Parsing Error: {str(e)}")
        answers = [f"Error: Failed to parse Gemini response. {str(e)}"]

    # === Cleanup ===
    try:
        os.remove(pdf_path)
    except Exception:
        pass

    return {"answers": answers}
