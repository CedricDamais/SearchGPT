# Re-ranking Methods Comparison

## Overview: 4 Approaches to Compare

| Method | Description | Training Required | Memory | Latency | Cost | Best For |
|--------|-------------|-------------------|--------|---------|------|----------|
| **Prompt-Based LLM** | Use GPT/Claude/Ollama with prompts | ❌ No | Low | High (1-2s) | High ($$$) | Prototyping, low volume |
| **LoRA Fine-tuned** | Fine-tune with LoRA adapters | ✅ Yes (~6h) | High (16GB) | Medium (50-100ms) | Medium ($$) | Custom domain, quality focus |
| **Quantized (INT4)** | Post-training quantization | ❌ No | Very Low (4GB) | Low (20-40ms) | Very Low ($) | Production, cost-sensitive |
| **QLoRA Fine-tuned** | Train LoRA on quantized model | ✅ Yes (~8h) | Low (8GB) | Low (20-40ms) | Low ($) | **Best overall balance** |

---

## Detailed Comparison

### 1️⃣ Prompt-Based Re-ranking (Baseline)

**How it works:**
```python
prompt = f"""Rate relevance (0-1):
Query: {query}
Document: {document}
Score:"""

response = ollama.generate(model="llama3", prompt=prompt)
score = extract_number(response)
```

**Pros:**
- ✅ Zero setup, works immediately
- ✅ Very flexible (change prompts easily)
- ✅ Can leverage latest LLMs (GPT-4, Claude)
- ✅ Good for prototyping

**Cons:**
- ❌ Slow (1-2 seconds per document)
- ❌ Expensive (API costs add up)
- ❌ Inconsistent outputs (parsing issues)
- ❌ Not suitable for production scale

**When to use:**
- Quick MVP/demo
- Low query volume (<100/day)
- Experimental phase

---

### 2️⃣ LoRA Fine-Tuned Model

**How it works:**
```python
# Training phase
base_model = "llama-3-8b"
lora_config = LoraConfig(r=16, lora_alpha=32)
model = get_peft_model(base_model, lora_config)
train(model, msmarco_dataset)  # ~6 hours on A100

# Inference
inputs = tokenizer(f"{query} [SEP] {document}")
score = model(inputs).logits[1]  # Probability of "relevant"
```

**Pros:**
- ✅ High quality (trained on your data)
- ✅ Fast inference (50-100ms)
- ✅ Small adapter size (~20MB)
- ✅ Can train on specific domain

**Cons:**
- ❌ Requires good GPU (24GB VRAM)
- ❌ Training takes several hours
- ❌ Still uses 16GB for inference
- ❌ Need training data

**When to use:**
- You have training data
- Quality is priority
- Have access to good GPU
- Custom domain (medical, legal, etc.)

**Training Requirements:**
- GPU: A100 (40GB) or RTX 3090 (24GB)
- Time: 4-6 hours for 100k examples
- Data: 50k-500k query-doc pairs

---

### 3️⃣ Post-Training Quantization (INT8/INT4)

**How it works:**
```python
# Option A: Load pre-trained with quantization
model = AutoModel.from_pretrained(
    "cross-encoder/ms-marco-MiniLM",
    load_in_4bit=True  # Automatic quantization
)

# Option B: Quantize your LoRA model
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_pretrained("./lora_model")
model.quantize(bits=4)
model.save_quantized("./lora_model_int4")
```

**Pros:**
- ✅ 4x smaller model size (14GB → 3.5GB)
- ✅ 2-3x faster inference
- ✅ Runs on smaller GPUs (8GB)
- ✅ No training required
- ✅ Minimal quality loss (1-2%)

**Cons:**
- ❌ Quality degrades slightly
- ❌ Can't improve beyond base model
- ❌ Quantization can be tricky

**When to use:**
- Need to deploy existing model
- Memory/cost constraints
- Can accept 1-2% quality loss
- Production deployment

**Quantization Methods:**
| Method | Precision | Memory | Quality Loss | Speed |
|--------|-----------|--------|--------------|-------|
| **bitsandbytes INT8** | 8-bit | 50% | <0.5% | 1.5x faster |
| **GPTQ INT4** | 4-bit | 75% | 1-2% | 2-3x faster |
| **AWQ INT4** | 4-bit | 75% | 0.5-1% | 2-3x faster |

---

### 4️⃣ QLoRA Fine-Tuning (🏆 Recommended)

**How it works:**
```python
# Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModel.from_pretrained(
    "llama-3-8b",
    quantization_config=bnb_config
)

# Apply LoRA on top of quantized model
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=16, lora_alpha=32)
model = get_peft_model(model, lora_config)

# Train (uses only 6-8GB VRAM!)
train(model, msmarco_dataset)
```

**Pros:**
- ✅ Best of both worlds: quality + efficiency
- ✅ Train on consumer GPU (8GB VRAM)
- ✅ Fast inference (20-40ms)
- ✅ Small memory footprint
- ✅ 99% of full LoRA quality
- ✅ Much cheaper training

**Cons:**
- ❌ Training takes slightly longer (~20% slower)
- ❌ Requires good understanding of both techniques
- ❌ Need BF16-compatible GPU (Ampere+)

**When to use:**
- Limited GPU memory (<16GB)
- Want to fine-tune but can't afford big GPU
- Production deployment after training
- **This should be your main approach!**

**Training Requirements:**
- GPU: RTX 3090 (24GB), RTX 4090 (24GB), or even RTX 3060 (12GB)
- Time: 6-10 hours for 100k examples
- Data: 50k-500k query-doc pairs

---

## 🎯 Recommended Workflow for Your CV Project

### **Phase 1: Baseline (Week 1)**
```
1. Implement prompt-based re-ranking (Ollama)
2. Benchmark on MS MARCO (100 queries)
3. Measure: latency, NDCG@10
4. Document limitations
```

### **Phase 2: LoRA Fine-tuning (Week 2)**
```
1. Prepare MS MARCO dataset (100k pairs)
2. Fine-tune Llama-3-8B with LoRA
3. Compare vs. prompt-based
4. Document: training curves, final metrics
```

### **Phase 3: Quantization (Week 3)**
```
1. Apply INT8 quantization to LoRA model
2. Apply INT4 quantization (GPTQ)
3. Benchmark all variants (FP16, INT8, INT4)
4. Create comparison charts
```

### **Phase 4: QLoRA (Week 4)**
```
1. Train QLoRA from scratch
2. Compare: LoRA vs QLoRA (training time, memory, quality)
3. Deploy quantized model
4. Write comprehensive report
```

---

## 📊 Expected Results

### Performance Metrics
| Method | NDCG@10 | Latency (ms) | Memory (GB) | Cost/1M queries |
|--------|---------|--------------|-------------|-----------------|
| Prompt-based (GPT-4) | 0.875 | 1500 | 2 | $300 |
| Prompt-based (Ollama) | 0.820 | 800 | 8 | $0 |
| LoRA Fine-tuned (FP16) | 0.865 | 45 | 16 | $0 |
| LoRA + INT8 Quantized | 0.860 | 35 | 8 | $0 |
| LoRA + INT4 Quantized | 0.850 | 28 | 4 | $0 |
| QLoRA Fine-tuned + INT4 | 0.860 | 28 | 4 | $0 |

### Training Comparison
| Method | GPU Required | VRAM (GB) | Training Time | Adapter Size |
|--------|--------------|-----------|---------------|--------------|
| Full Fine-tuning | A100 | 40 | 10h | 14GB |
| LoRA | A100/3090 | 24 | 6h | 20MB |
| QLoRA | 3090/4090 | 8 | 8h | 20MB |

---

## 🚀 Tools & Libraries

```bash
# Core dependencies
uv add torch transformers

# LoRA/QLoRA
uv add peft accelerate

# Quantization
uv add bitsandbytes auto-gptq

# Training utilities
uv add datasets wandb tqdm

# Serving
uv add vllm fastapi

# Evaluation
uv add scikit-learn numpy pandas matplotlib
```

---

## 📝 Deliverables for CV

Create these artifacts:

1. **Training Notebooks**
   - `notebooks/01_lora_training.ipynb`
   - `notebooks/02_quantization.ipynb`
   - `notebooks/03_qlora_training.ipynb`

2. **Benchmark Reports**
   - `results/method_comparison.md`
   - `results/quantization_ablation.md`
   - `results/lora_vs_qlora.md`

3. **Visualizations**
   - Training curves
   - Latency vs Quality scatter plot
   - Memory usage bar chart

4. **Blog Post**
   - "Fine-tuning LLMs for Search: LoRA vs QLoRA"
   - Post on Medium/Dev.to
   - Share on LinkedIn

---

## 🎓 Key Learnings to Highlight

In interviews, you can discuss:

1. **Trade-offs**: Quality vs Speed vs Memory vs Cost
2. **When to use each method**: Based on constraints
3. **Quantization impact**: How 4-bit affects different layer types
4. **Training efficiency**: Why QLoRA enables consumer GPU training
5. **Production deployment**: Serving optimizations (batching, KV cache)

---

**This comprehensive approach will make you stand out from 99% of ML engineer candidates!** 🚀
