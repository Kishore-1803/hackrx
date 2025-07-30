from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import tempfile
import os
import asyncio
import time

from agno.agent import Agent
from agno.models.google import Gemini
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.embedder.google import GeminiEmbedder

# === FastAPI App Initialization ===
app = FastAPI()

# === API Keys ===
GOOGLE_API_KEY = "AIzaSyCAZN6O2VZeLappQbR-gDCgaimKp0AgVNM"  # flash
GOOGLE_API_KEY_2 = "AIzaSyC-1DsPyxFMa0NNPa6cViSZxs0Ypq4qx0E"  # embedder

# === Rate Limiting ===
last_request_time = 0
REQUEST_INTERVAL = 12  # 12 seconds between requests (5 requests/minute)

# === PgVector Setup ===
db_url = "postgresql://postgres:Kishsiva%40123@db.ugxvebypwxhcgrvlhgad.supabase.co:5432/postgres"
vector_db = PgVector(
    table_name="hackrx_policy_documents",
    db_url=db_url,
    embedder=GeminiEmbedder(api_key=GOOGLE_API_KEY_2),
)

# === Pydantic Models ===
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# === Rate Limited Helper Function ===
async def get_answer_with_rate_limit(agent, question: str, retries: int = 3, timeout: int = 60):
    global last_request_time
    
    for attempt in range(retries):
        try:
            # Rate limiting: ensure 12 seconds between requests
            current_time = time.time()
            time_since_last = current_time - last_request_time
            if time_since_last < REQUEST_INTERVAL:
                wait_time = REQUEST_INTERVAL - time_since_last
                await asyncio.sleep(wait_time)
            
            last_request_time = time.time()
            
            # Execute with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(agent.run, question), 
                timeout=timeout
            )
            return response
            
        except asyncio.TimeoutError:
            if attempt < retries - 1:
                await asyncio.sleep(5)  # Wait before retry
            else:
                return "Error: Request timed out after 60 seconds"
                
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "429" in err or "503" in err:
                if attempt < retries - 1:
                    wait_time = 30 * (attempt + 1)  # 30, 60, 90 seconds
                    await asyncio.sleep(wait_time)
                else:
                    return "Error: Gemini rate limit exceeded. Please try again later."
            else:
                return f"Error: {err}"
    
    return "Error: All retry attempts failed"

# === POST Endpoint ===
@app.post("/hackrx/run", response_model=QueryResponse)
async def ask_document_questions(
    body: QueryRequest,
    authorization: str = Header(...)
):
    # === Bearer Token Validation ===
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    # === Download PDF to temp file ===
    try:
        response = requests.get(body.documents, timeout=30)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            pdf_path = temp_pdf.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}")

    # === Vector DB and Agent Setup ===
    try:
        knowledge_base = PDFKnowledgeBase(path=pdf_path, vector_db=vector_db)
        knowledge_base.load(recreate=False, upsert=True)

        agent = Agent(
            model=Gemini(id="gemini-2.5-flash", api_key=GOOGLE_API_KEY),
            knowledge=knowledge_base,
            show_tool_calls=False,
            markdown=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB error: {str(e)}")

    # === Ask Questions with Rate Limiting ===
    answers = []
    for i, question in enumerate(body.questions):
        try:
            print(f"Processing question {i+1}/{len(body.questions)}: {question[:50]}...")
            response = await get_answer_with_rate_limit(agent, question)
            
            if isinstance(response, str):
                answers.append(response)
            else:
                answers.append(response.content.strip())
                
        except Exception as e:
            answers.append(f"Error: {str(e)}")

    # === Cleanup ===
    try:
        os.remove(pdf_path)
    except Exception:
        pass

    return {"answers": answers}