# SearchGPT 🔍

LLM-powered search engine with hybrid search and re-ranking capabilities.

## Features

- 🔄 **Hybrid Search**: Combines FAISS and ElasticSearch for optimal results
- 🤖 **LLM Re-ranking**: Uses large language models to re-rank search results for better relevance
- ⚡ **FastAPI Backend**: High-performance REST API
- 📊 **Evaluation Metrics**: Built-in support for NDCG, MRR, and other IR metrics
- 🧪 **Comprehensive Testing**: Full test suite with pytest

## Project Structure

```
SearchGPT/
├── src/
│   ├── api/              # FastAPI application
│   ├── hybrid_search/    # Hybrid search implementation
│   ├── llm_reranking/    # LLM re-ranking logic
│   ├── evaluation/       # Metrics and benchmarks
│   ├── core/            # Utilities (config, logging, cache)
│   └── deployment/      # Docker and deployment files
├── tests/               # Test suite
├── scripts/             # Utility scripts
├── data/               # Data directory (indices, embeddings)
└── resources/          # Research papers and documentation
```

## Getting Started

### Prerequisites

- Python 3.9+
- [UV](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/SearchGPT.git
   cd SearchGPT
   ```

2. **Install dependencies with UV**
   ```bash
   uv sync
   ```
   
   Or with pip:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

### Running the API

```bash
uv run uvicorn src.api.main:app --reload

uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive API documentation.

### Running Tests

```bash
uv run pytest

pytest

pytest --cov=src --cov-report=html
```

## API Usage

### Search Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does hybrid search work?",
    "top_k": 10,
    "use_reranking": true,
    "hybrid_alpha": 0.5
  }'
```

### Response

```json
{
  "query": "How does hybrid search work?",
  "results": [
    {
      "id": "doc1",
      "title": "Introduction to Hybrid Search",
      "content": "Hybrid search combines...",
      "score": 0.95,
      "metadata": {}
    }
  ],
  "total": 1,
  "processing_time_ms": 123.45
}
```

## Configuration

Configuration is managed through environment variables (see `.env.example`):

- `OPENAI_API_KEY`: OpenAI API key for embeddings and re-ranking
- `DEFAULT_LLM_MODEL`: LLM model to use (default: gpt-4o-mini)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)
- `DEFAULT_TOP_K`: Number of results to return (default: 10)
- `DEFAULT_HYBRID_ALPHA`: Balance between BM25 (0.0) and vector (1.0) search

## Development

### Code Formatting

```bash
uv run black src tests

uv run ruff check src tests
```

### Adding Dependencies

```bash
uv add package-name

uv add --dev package-name
```

## Scripts

- `scripts/setup_indices.py`: Initialize search indices
- `scripts/run_benchmark.py`: Run evaluation benchmarks

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Resources

Research papers and documentation can be found in the `resources/` directory.
