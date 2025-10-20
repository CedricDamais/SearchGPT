"""
Streamlit frontend for SearchGPT - LLM-powered search engine.
"""

import streamlit as st
import requests
import time
import json
from typing import List, Dict, Any
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
SEARCH_ENDPOINT = f"{API_BASE_URL}/api/v1/search"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Page configuration
st.set_page_config(
    page_title="SearchGPT",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
def load_css():
    """Load custom CSS for better UI."""
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    
    .search-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .result-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    
    .result-score {
        float: right;
        background: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    
    .result-content {
        color: #666;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    
    .result-metadata {
        font-size: 0.8rem;
        color: #888;
    }
    
    .stats-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #fff5f5;
        color: #c53030;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #c53030;
    }
    
    .success-message {
        background: #f0fff4;
        color: #38a169;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #38a169;
    }
    </style>
    """, unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def perform_search(query: str, top_k: int, use_reranking: bool, hybrid_alpha: float) -> Dict[str, Any]:
    """Perform search via API."""
    payload = {
        "query": query,
        "top_k": top_k,
        "use_reranking": use_reranking,
        "hybrid_alpha": hybrid_alpha,
    }
    
    try:
        response = requests.post(SEARCH_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")

def display_search_result(result: Dict[str, Any], index: int):
    """Display a single search result."""
    with st.container():
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">
                {result.get('title', 'No Title')}
                <span class="result-score">Score: {result.get('score', 0):.3f}</span>
            </div>
            <div class="result-content">
                {result.get('content', 'No content available')[:300]}...
            </div>
            <div class="result-metadata">
                ID: {result.get('id', 'N/A')} | 
                Metadata: {json.dumps(result.get('metadata', {})) if result.get('metadata') else 'None'}
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application."""
    load_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 SearchGPT</h1>
        <p>LLM-powered search engine with hybrid search and re-ranking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Search Settings")
        
        # API Health Check
        if check_api_health():
            st.markdown('<div class="success-message">✅ API is running</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">❌ API is not accessible</div>', unsafe_allow_html=True)
            st.warning(f"Make sure the API is running at {API_BASE_URL}")
        
        st.markdown("---")
        
        # Search parameters
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum number of search results to return"
        )
        
        use_reranking = st.checkbox(
            "Enable LLM Re-ranking",
            value=True,
            help="Use LLM to re-rank search results for better relevance"
        )
        
        hybrid_alpha = st.slider(
            "Hybrid Search Balance",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0.0 = Pure BM25, 1.0 = Pure Vector Search"
        )
        
        st.markdown("---")
        
        # Search history (stored in session state)
        if "search_history" not in st.session_state:
            st.session_state.search_history = []
        
        if st.session_state.search_history:
            st.subheader("🕒 Recent Searches")
            for i, hist_query in enumerate(st.session_state.search_history[-5:]):
                if st.button(f"'{hist_query[:30]}...'", key=f"hist_{i}"):
                    st.session_state.current_query = hist_query
                    st.rerun()
    
    # Main search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get current query from session state or use empty string
        current_query = st.session_state.get("current_query", "")
        
        query = st.text_input(
            "Enter your search query:",
            value=current_query,
            placeholder="e.g., How does hybrid search work?",
            help="Enter your search query and press Enter or click Search"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Perform search
    if (search_button or query) and query.strip():
        # Add to search history
        if query not in st.session_state.search_history:
            st.session_state.search_history.append(query)
        
        # Clear current_query from session state
        if "current_query" in st.session_state:
            del st.session_state.current_query
        
        with st.spinner("Searching..."):
            try:
                start_time = time.time()
                results = perform_search(query, top_k, use_reranking, hybrid_alpha)
                search_time = time.time() - start_time
                
                # Display results statistics
                st.markdown(f"""
                <div class="stats-container">
                    📊 <strong>Search Results</strong><br>
                    Query: "<em>{query}</em>"<br>
                    Found {results.get('total', 0)} results in {search_time:.2f} seconds<br>
                    API Processing Time: {results.get('processing_time_ms', 0):.1f} ms
                </div>
                """, unsafe_allow_html=True)
                
                # Display results
                if results.get('results'):
                    for i, result in enumerate(results['results']):
                        display_search_result(result, i)
                else:
                    st.info("No results found. Try a different query or adjust the search parameters.")
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-message">
                    <strong>Search Error:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)
    
    if not query:
        st.markdown("---")
        st.markdown("### Example Queries")
        
        col1, col2, col3 = st.columns(3)
        
        example_queries = [
            "How does machine learning work?",
            "What is natural language processing?",
            "Explain deep learning algorithms",
            "Vector embeddings in search",
            "BM25 ranking algorithm",
            "Hybrid search techniques"
        ]
        
        for i, example in enumerate(example_queries):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(f"🔍 {example}", key=f"example_{i}"):
                    st.session_state.current_query = example
                    st.rerun()

if __name__ == "__main__":
    main()