"""Search endpoints."""

from fastapi import APIRouter, HTTPException
from src.api.models.request import SearchRequest
from src.api.models.response import SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform hybrid search with LLM re-ranking.
    
    Args:
        request: Search request containing query and optional parameters
        
    Returns:
        SearchResponse with ranked results
    """
    try:
        # TODO: Implement actual search logic
        # 1. Run hybrid search (BM25 + vector search)
        # 2. Apply LLM re-ranking
        # 3. Return ranked results
        
        return SearchResponse(
            query=request.query,
            results=[],
            total=0,
            processing_time_ms=0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
