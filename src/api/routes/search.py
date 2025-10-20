"""Search routes for the API."""

from typing import List
from fastapi import APIRouter, HTTPException, Query

from src.api.models.request import SearchRequest
from src.api.models.response import SearchResponse, SearchResult
from src.core.search_manager import get_search_manager
from src.core.logging import logger

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Perform search using the configured search manager.
    
    Args:
        request: Search request with query and parameters
        
    Returns:
        Search response with results
    """
    try:
        search_manager = get_search_manager()

        logger.info(f"Searching for: {request.query}")
        raw_results = search_manager.get_search_results(
            query=request.query,
            top_k=request.top_k,
            use_hybrid=True,
            hybrid_alpha=request.hybrid_alpha or 0.5
        )
        logger.info(f"Search returned {len(raw_results)} results for query: {request.query}")
        
        results = [
            SearchResult(
                id=result["id"],
                title=result["title"],
                content=result["content"],
                score=result["score"],
                metadata=result["metadata"]
            )
            for result in raw_results
        ]
        
        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
            processing_time_ms=0.0  # TODO: Add timing
        )
        
    except RuntimeError as e:
        if "not initialized" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Search indices not initialized. Please run setup_indices.py first."
            )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search", response_model=SearchResponse)
async def search_get(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results to return"),
    use_reranking: bool = Query(True, description="Use LLM re-ranking"),
    hybrid_alpha: float = Query(0.5, ge=0.0, le=1.0, description="Hybrid search balance")
) -> SearchResponse:
    """
    Perform search using GET request with query parameters.
    
    Args:
        q: Search query
        top_k: Number of results to return
        use_reranking: Whether to use LLM re-ranking
        hybrid_alpha: Balance between BM25 and vector search
        
    Returns:
        Search response with results
    """
    request = SearchRequest(
        query=q,
        top_k=top_k,
        use_reranking=use_reranking,
        hybrid_alpha=hybrid_alpha
    )
    return await search(request)


@router.get("/search/stats")
async def search_stats():
    """Get search manager statistics."""
    try:
        search_manager = get_search_manager()
        stats = search_manager.get_stats()
        return {
            "status": "ready" if stats["initialized"] else "not_initialized",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
