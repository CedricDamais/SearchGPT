# SearchGPT: ML Engineer Portfolio Project Roadmap

## 🎯 Project Goal
Build a production-ready hybrid search engine demonstrating:
- Classical & modern ML techniques
- System design & engineering skills
- Rigorous evaluation methodology
- End-to-end deployment capabilities

---

## 📅 Implementation Roadmap (4-5 weeks)

### **Phase 1: Hybrid Search Foundation** (Week 1-2)

#### 1.1 BM25 Implementation
- [ ] Implement BM25 algorithm from scratch (or use Rank-BM25)
- [ ] Build inverted index for documents
- [ ] Add tokenization & preprocessing
- [ ] Test with sample queries
- **CV Highlight**: "Implemented classical information retrieval with BM25 ranking"

**Files to create:**
- `src/hybrid_search/bm25.py`
- `src/hybrid_search/preprocessing.py`
- `tests/test_hybrid_search/test_bm25.py`

#### 1.2 Vector Search Implementation
- [ ] Integrate embedding model (Ollama or Sentence-Transformers)
- [ ] Build FAISS vector index
- [ ] Implement semantic search
- [ ] Add batch embedding generation
- **CV Highlight**: "Built semantic search using dense embeddings and FAISS"

**Files to create:**
- `src/hybrid_search/vector_search.py`
- `src/hybrid_search/embeddings.py`
- `tests/test_hybrid_search/test_vector_search.py`

#### 1.3 Hybrid Fusion
- [ ] Implement Reciprocal Rank Fusion (RRF)
- [ ] Add weighted fusion (alpha parameter)
- [ ] Compare fusion strategies
- **CV Highlight**: "Designed hybrid retrieval system combining lexical and semantic search"

**Files to create:**
- `src/hybrid_search/fusion.py`
- `tests/test_hybrid_search/test_fusion.py`

---

### **Phase 2: LLM Re-ranking** (Week 3)

#### 2.1 Prompt-Based Re-ranking (Already Started!)
- [x] Basic implementation in `ollama_client.py`
- [ ] Optimize prompts for better scoring
- [ ] Add temperature/sampling controls
- [ ] Handle batch processing efficiently
- **CV Highlight**: "Integrated LLM-based re-ranking using prompt engineering"

#### 2.2 Fine-Tuned Re-ranker with LoRA
- [ ] Prepare training dataset (MS MARCO query-document pairs)
- [ ] Fine-tune base model (e.g., Llama-3-8B or Mistral-7B) using LoRA
- [ ] Use PEFT library for parameter-efficient fine-tuning
- [ ] Track training metrics (loss, validation accuracy)
- [ ] Save LoRA adapters
- **CV Highlight**: "Fine-tuned 7B parameter LLM using LoRA (Low-Rank Adaptation) for document re-ranking"

**Files to create:**
- `src/llm_reranking/fine_tuned/lora_trainer.py`
- `src/llm_reranking/fine_tuned/dataset_loader.py`
- `src/llm_reranking/fine_tuned/inference.py`
- `training_configs/lora_config.yaml`

#### 2.3 Model Quantization
- [ ] Apply 8-bit quantization using bitsandbytes
- [ ] Apply 4-bit quantization (GPTQ or AWQ)
- [ ] Benchmark: FP16 vs INT8 vs INT4
- [ ] Measure latency, memory, accuracy trade-offs
- [ ] Document compression ratios and performance
- **CV Highlight**: "Optimized model inference with 4-bit quantization, reducing memory by 75% while maintaining 98% accuracy"

**Files to create:**
- `src/llm_reranking/fine_tuned/quantization.py`
- `scripts/quantize_model.py`
- `benchmarks/quantization_results.md`

#### 2.4 QLoRA Fine-Tuning
- [ ] Implement QLoRA (Quantized LoRA) training
- [ ] Fine-tune on 4-bit quantized base model
- [ ] Compare: Full LoRA vs QLoRA (speed, memory, quality)
- [ ] Document memory savings (e.g., 24GB → 8GB VRAM)
- [ ] Create deployment pipeline
- **CV Highlight**: "Implemented QLoRA fine-tuning, enabling 7B model training on consumer GPU with 4x memory reduction"

**Files to create:**
- `src/llm_reranking/fine_tuned/qlora_trainer.py`
- `training_configs/qlora_config.yaml`
- `docs/qlora_training_guide.md`

#### 2.5 Production Deployment
- [ ] Package quantized model for serving
- [ ] Set up vLLM or Text Generation Inference (TGI)
- [ ] Implement batching and KV cache optimization
- [ ] Add model version management
- [ ] Create A/B testing infrastructure
- **CV Highlight**: "Deployed quantized LLM with vLLM, achieving 10x throughput improvement"

**Files to create:**
- `src/deployment/model_server.py`
- `docker/quantized-model.Dockerfile`
- `deployment_configs/vllm_config.yaml`

#### 2.3 Comparative Analysis
- [ ] Benchmark: prompt-based vs fine-tuned
- [ ] Latency comparison
- [ ] Quality comparison
- [ ] Cost analysis
- **CV Highlight**: "Conducted comprehensive evaluation of re-ranking approaches"

---

### **Phase 3: Evaluation & Metrics** (Week 4)

#### 3.1 Implement IR Metrics
- [ ] NDCG@k (Normalized Discounted Cumulative Gain)
- [ ] MRR (Mean Reciprocal Rank)
- [ ] MAP (Mean Average Precision)
- [ ] Precision@k, Recall@k
- **CV Highlight**: "Implemented standard IR evaluation metrics (NDCG, MRR, MAP)"

**Files to create:**
- `src/evaluation/metrics.py`
- `tests/test_evaluation/test_metrics.py`

#### 3.2 Create Benchmark Dataset
- [ ] Download MS MARCO dev set (or use BEIR)
- [ ] Create smaller test set (100-500 queries)
- [ ] Format as JSON for easy loading
- [ ] Document dataset statistics
- **CV Highlight**: "Curated evaluation dataset with 500+ query-document pairs"

**Files to create:**
- `src/evaluation/datasets.py`
- `data/datasets/test_queries.json`
- `data/datasets/README.md`

#### 3.3 Benchmarking Framework
- [ ] Run all methods on test set
- [ ] Generate comparison tables
- [ ] Create visualizations (matplotlib/plotly)
- [ ] Write analysis report
- **CV Highlight**: "Built automated benchmarking pipeline with visualization"

**Files to create:**
- `src/evaluation/benchmark_runner.py`
- `scripts/run_full_evaluation.py`
- `results/benchmark_report.md`

---

### **Phase 4: Production & Deployment** (Week 5)

#### 4.1 Performance Optimization
- [ ] Add Redis caching for frequent queries
- [ ] Implement query result caching
- [ ] Add request batching
- [ ] Profile & optimize bottlenecks
- **CV Highlight**: "Optimized search latency by 10x using caching and batching"

**Files to create:**
- `src/core/redis_cache.py`
- `src/api/middleware/caching.py`

#### 4.2 Monitoring & Observability
- [ ] Add Prometheus metrics
- [ ] Log query latencies
- [ ] Track error rates
- [ ] Create health checks
- **CV Highlight**: "Implemented monitoring with Prometheus metrics"

**Files to create:**
- `src/api/monitoring.py`
- `src/api/middleware/metrics.py`

#### 4.3 Docker & Deployment
- [ ] Optimize Dockerfile (multi-stage build)
- [ ] Add docker-compose.yml (with Redis)
- [ ] Create deployment docs
- [ ] Add CI/CD pipeline (GitHub Actions)
- **CV Highlight**: "Containerized ML service with Docker and CI/CD pipeline"

**Files to create:**
- `docker-compose.yml`
- `.github/workflows/ci.yml`
- `docs/deployment.md`

#### 4.4 Documentation & Demo
- [ ] Create comprehensive README
- [ ] Add API documentation
- [ ] Record demo video
- [ ] Write technical blog post
- **CV Highlight**: "Documented and presented system architecture and design decisions"

---

## 🎯 Key Deliverables for CV

### **1. GitHub Repository**
- Clean, well-organized code
- Comprehensive README with architecture diagram
- Full test coverage (aim for >80%)
- Professional documentation

### **2. Technical Report/Blog Post**
Write about:
- System architecture
- Why hybrid search works
- Re-ranking comparison (prompt vs fine-tuned)
- Performance benchmarks
- Lessons learned

### **3. Demo/Presentation**
- Live API demo
- Jupyter notebook with examples
- Performance visualizations
- Architecture diagram

---

## 📊 Skills Demonstrated

| Skill Category | Specific Skills |
|----------------|----------------|
| **ML Fundamentals** | Embeddings, transformers, fine-tuning, evaluation metrics |
| **Classical ML** | BM25, TF-IDF, information retrieval |
| **Deep Learning** | BERT, cross-encoders, semantic search |
| **ML Engineering** | API design, caching, optimization, deployment |
| **Data Engineering** | Vector databases, indexing, batch processing |
| **Software Engineering** | Testing, documentation, CI/CD, Docker |
| **System Design** | Hybrid architectures, latency optimization |
| **Evaluation** | A/B testing, metrics implementation, benchmarking |

---

## � Deep Dive: LoRA → Quantization → QLoRA Pipeline

### **Step 1: LoRA Fine-Tuning**
```python
# Training configuration
Base Model: Llama-3-8B or Mistral-7B
Method: LoRA (rank=16, alpha=32)
Dataset: MS MARCO passages (100k pairs)
Task: Binary classification (relevant/not relevant)
Hardware: Single GPU (24GB VRAM)
Training time: ~4-6 hours
```

**Key Metrics to Track:**
- Training loss curve
- Validation accuracy
- NDCG@10 improvement
- LoRA adapter size (typically ~10-50MB)

### **Step 2: Post-Training Quantization**
```python
# Quantization experiments
1. FP16 (baseline) → 16GB memory
2. INT8 (bitsandbytes) → 8GB memory  
3. INT4 (GPTQ/AWQ) → 4GB memory

Measure for each:
- Inference latency (ms/query)
- Memory usage (GB)
- NDCG@10 (quality check)
- Throughput (queries/sec)
```

### **Step 3: QLoRA Fine-Tuning** 
```python
# Train on quantized base model
Base: 4-bit quantized Llama-3-8B
Method: QLoRA (LoRA on quantized model)
Memory: ~8GB VRAM (vs 24GB for full LoRA)
Quality: Within 1-2% of full LoRA
Benefit: Fine-tune on consumer GPU!
```

### **Step 4: Production Deployment**
```python
# Serving stack
1. Model: 4-bit quantized + LoRA adapters
2. Server: vLLM or TGI
3. Optimizations:
   - Continuous batching
   - PagedAttention (KV cache)
   - Flash Attention 2
4. Target: <100ms latency, 100+ QPS
```

### **Expected CV Impact:**
This pipeline demonstrates:
- ✅ **Advanced fine-tuning** (LoRA/PEFT)
- ✅ **Model compression** (quantization)
- ✅ **Efficient training** (QLoRA)
- ✅ **Production optimization** (vLLM)
- ✅ **Experimental rigor** (A/B testing, benchmarking)

---

## �💡 Bonus Features (If Time)

1. **Query Understanding**
   - Query expansion
   - Spell correction
   - Intent classification

2. **Advanced Techniques**
   - Query rewriting with LLM
   - Pseudo-relevance feedback
   - Learning to rank

3. **UI/Frontend**
   - Simple web interface (React/Streamlit)
   - Search results visualization
   - Interactive demo

---

## 📝 CV Bullet Points You'll Be Able to Write

**Search & Retrieval:**
- "Built production-ready hybrid search engine combining BM25 and dense embeddings, achieving 15% improvement in NDCG@10"
- "Designed evaluation framework with standard IR metrics (NDCG, MRR, MAP) on 500+ query test set"

**LLM Fine-Tuning & Optimization:**
- "Fine-tuned 7B parameter LLM using LoRA (Low-Rank Adaptation) for document re-ranking, improving relevance by 20%"
- "Implemented QLoRA training pipeline, reducing GPU memory requirements by 75% (24GB → 6GB)"
- "Applied 4-bit quantization (GPTQ) to compress model by 75% while maintaining 98% of original accuracy"
- "Conducted ablation study comparing LoRA vs QLoRA vs full fine-tuning across latency, memory, and quality metrics"

**ML Engineering & Deployment:**
- "Deployed quantized LLM with vLLM serving framework, achieving <100ms p95 latency and 100+ QPS throughput"
- "Implemented A/B testing infrastructure to compare prompt-based vs fine-tuned re-ranking approaches"
- "Built containerized ML service with FastAPI, Redis caching, and Prometheus monitoring"
- "Achieved >85% test coverage with comprehensive unit and integration tests"

**Tools & Technologies:**
- PyTorch, Transformers, PEFT (LoRA/QLoRA), bitsandbytes, GPTQ, vLLM
- FAISS, BM25, Sentence-Transformers, Ollama
- FastAPI, Docker, Redis, Prometheus, pytest

---

## 🚀 Getting Started

Let's start with **Phase 1.1: BM25 Implementation**

Would you like me to:
1. Implement the BM25 search module?
2. Set up the data loading infrastructure?
3. Create the evaluation framework first?

Which phase excites you most? Let's prioritize what will be most impressive!
