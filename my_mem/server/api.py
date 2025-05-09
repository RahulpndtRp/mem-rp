"""
FastAPI wrapper around Memory + RAG.

Run with:
    uvicorn my_mem.api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from my_mem.configs.base import MemoryConfig
from my_mem.memory.main  import Memory
from my_mem.rag.rag_pipeline import RAGPipeline

# ---------- bootstrap singletons ------------------------------------------
_mem = Memory(MemoryConfig())
_rag = RAGPipeline(_mem, top_k=5)

app = FastAPI(
    title="my_mem RAG API",
    description="Long- + short-term memory store with RAG answering & citations",
    version="0.1.0",
)

# ---------- request / response models -------------------------------------
class AddReq(BaseModel):
    text: str
    user_id: str
    infer: bool = True


class SearchReq(BaseModel):
    query: str
    user_id: str
    limit: int = 5


class RagReq(BaseModel):
    question: str
    user_id: str
    top_k: int = 5


class RagResp(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


# ---------- routes --------------------------------------------------------
@app.post("/mem/add")
def add_memory(req: AddReq):
    return _mem.add(req.text, user_id=req.user_id, infer=req.infer)


@app.post("/mem/search")
def search_memory(req: SearchReq):
    return _mem.search(req.query, user_id=req.user_id, limit=req.limit)


@app.post("/rag/query", response_model=RagResp)
def rag_query(req: RagReq):
    try:
        rag = RAGPipeline(_mem, top_k=req.top_k)
        return rag.query(req.question, user_id=req.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
