import spaces
import gradio as gr
import pandas as pd

@spaces.GPU
def gpu_check():
    return "ZeroGPU is active"

# ==================== DATA ====================

results_data = {
    "Metric": ["Hit@1", "Hit@3", "MRR", "Faithfulness", "Answer Relevancy"],
    "Semantic": [0.900, 1.000, 0.950, 1.000, 0.950],
    "Parent-Child": [0.900, 1.000, 0.933, 1.000, 1.000],
    "Winner": ["Tie", "Tie", "Semantic ✓", "Tie", "Parent-Child ✓"]
}

df_results = pd.DataFrame(results_data)

# ==================== TAB 1: OVERVIEW ====================

def tab_overview():
    return """
# 🦙 RAGBench - Overview

## What is RAGBench?

RAGBench is a **controlled evaluation framework** for comparing different RAG chunking strategies:

- **Semantic/Document-Level Chunking** - Simple, document-based
- **Parent-Child Chunking** - Sophisticated, hierarchical

Using the same corpus, queries, embedding model, and evaluation metrics.

## 🎯 Purpose

When building RAG systems, one critical question arises:

**"What's the best way to chunk documents?"**

- Chunks that are **too large** contain unnecessary information
- Chunks that are **too small** lose surrounding context

**RAGBench provides the answer with data-driven benchmarking.**

## 📊 Metrics Evaluated

### Retrieval Quality
- **Hit@1** - Is the relevant result ranked first?
- **Hit@3** - Is a relevant result in top 3?
- **MRR** - How high does the first relevant result rank?

### Generation Quality
- **Faithfulness** - Is the answer grounded in context?
- **Answer Relevancy** - Does the answer address the query?

---

**Separating evaluation into retrieval and generation helps identify 
where problems originate.**
"""

# ==================== TAB 2: ABOUT ====================

def tab_about():
    return """
# 🧪 About RAGBench

## The Problem

When building RAG systems, engineers choose chunking strategies based on:
- ❌ Gut feeling
- ❌ Blog posts
- ❌ Trial-and-error
- ❌ Random experimentation

**There was no standardized way to measure the impact.**

## The Solution

RAGBench provides:

✅ **Fixed Corpus** - 20 carefully curated RAG-focused documents
✅ **Fixed Queries** - 20 benchmark questions
✅ **Multiple Metrics** - Retrieval + generation evaluation
✅ **Reproducible** - Same setup for fair comparisons
✅ **Customizable** - Easy to extend and adapt

## 🏗️ How It Works

```
Query
  ↓
Chunking Strategy (Semantic or Parent-Child)
  ↓
Vector Retrieval (ChromaDB + Embeddings)
  ↓
Retrieved Context
  ↓
LLM Generation (Groq)
  ↓
RAGAS Evaluation
  ↓
Metrics (Hit@1, MRR, Faithfulness, Answer Relevancy)
```

## 🔧 Technologies Used

- **ChromaDB** - Vector database for semantic search
- **Sentence-Transformers** - Generate embeddings
- **RAGAS** - Evaluation framework for RAG systems
- **Groq API** - Fast LLM inference
- **Python** - All orchestration

## 📦 Project Structure

```
ragbench/
├── corpus.json                  # 20 evaluation documents
├── test_queries.json            # 20 benchmark queries  
├── eval.py                      # Run evaluation
├── analyze_results.py           # Compare strategies
├── app.py                       # This Gradio interface
├── requirements.txt             # Dependencies
│
└── results/
    ├── results_semantic.json
    ├── results_parent-child.json
    └── EVALUATION_REPORT.md
```
"""

# ==================== TAB 3: RESULTS ====================

def tab_results():
    results_html = df_results.to_html(index=False)
    
    return f"""
# 📈 Current Benchmark Results

## Semantic vs Parent-Child Comparison

{results_html}

## Key Findings

### Retrieval Performance

**Hit@1 & Hit@3:** Both strategies tie perfectly
- 90% of queries have relevant result at rank 1
- 100% of queries have relevant result in top 3

**MRR (Mean Reciprocal Rank):** Semantic wins slightly
- Semantic: 0.950
- Parent-Child: 0.933
- Difference: Semantic ranks relevant results ~2% higher

### Generation Performance

**Faithfulness:** Perfect tie
- Both strategies: 1.000
- Generated answers are fully grounded in context
- No hallucinations detected

**Answer Relevancy:** Parent-Child wins
- Semantic: 0.950
- Parent-Child: 1.000 (perfect)
- Parent-Child produces more directly relevant answers

## Query-Level Analysis

Across 20 benchmark queries:

| Outcome | Count | Percentage |
|---------|-------|-----------|
| Semantic wins | 0/20 | 0% |
| Parent-Child wins | 4/20 | 20% |
| Ties | 16/20 | 80% |

## Interpretation

**There is no universal winner.**

Each strategy excels at different metrics:

🟦 **Choose Semantic if:** 
- Better retrieval ranking matters most
- Simple implementation preferred
- Consistent performance needed

🟪 **Choose Parent-Child if:**
- Answer quality is priority
- Context preservation valued
- Handling complex multi-fact questions

**Conclusion:** Select based on your priorities, not arbitrary choice.
"""

# ==================== TAB 4: HOW TO USE ====================

def tab_how_to_use():
    return """
# 🚀 How to Use RAGBench

## Quick Start (5 minutes)

### 1. Clone Repository

```bash
git clone https://github.com/GulrezQayyum/ragbench.git
cd ragbench
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Groq API Key

Get free API key: https://console.groq.com

```bash
export GROQ_API_KEY="your_api_key_here"
```

Or create `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

### 4. Run Evaluation

```bash
python eval.py
```

Results saved to: `results/results_semantic.json` and `results/results_parent-child.json`

### 5. Analyze Results

```bash
python analyze_results.py
```

Full analysis in: `results/EVALUATION_REPORT.md`

---

## Customization

### Add More Evaluation Queries

Edit `test_queries.json`:

```json
{
  "query_id": "q_21",
  "query": "Your custom question?",
  "relevant_ids": ["doc_05"]
}
```

More queries = more reliable results.

### Change Embedding Model

In `eval.py`, modify:

```python
embeddings_model = SentenceTransformer("all-mpnet-base-v2")
```

Try: `all-MiniLM-L6-v2` (faster), `all-mpnet-base-v2` (better quality)

### Adjust Chunking Configuration

In `eval.py`, change parent-child parameters:

```python
sentences_per_child = 3  # Smaller = finer chunks
```

### Change Retrieval Top-K

In `eval.py`:

```python
top_k = 5  # Instead of 3
```

---

## Troubleshooting

**Issue:** `GROQ_API_KEY not set`
```bash
# Solution: Verify environment variable
echo $GROQ_API_KEY

# Or use .env file
cat .env
```

**Issue:** Slow evaluation
- Normal! LLM inference takes time (~2-3 min per strategy)
- Benchmark is lightweight intentionally

**Issue:** Module not found
```bash
# Verify all dependencies
pip install -r requirements.txt --upgrade
```
"""

# ==================== TAB 5: DOCS ====================

def tab_documentation():
    return """
# 📚 Complete Documentation

## Metrics Explained

### Retrieval Metrics

**Hit@K (Hit at K)**
- Measures if any relevant document appears in top-K results
- Hit@1: Relevant result is 1st
- Hit@3: Relevant result is in top 3
- Range: 0.0 to 1.0 (higher is better)

**Mean Reciprocal Rank (MRR)**
- Average of reciprocal ranks of first relevant results
- Formula: 1/N × Σ(1/rank of first relevant result)
- Range: 0.0 to 1.0 (higher is better)
- Example: If first relevant is at rank 2, contributes 0.5

### Generation Metrics

**Faithfulness**
- Does generated answer strictly follow retrieved context?
- Penalizes hallucinations and unsupported claims
- Range: 0.0 to 1.0 (higher is better)

**Answer Relevancy**
- Does answer directly address the question?
- Penalizes verbose or off-topic responses
- Range: 0.0 to 1.0 (higher is better)

---

## Why Multiple Metrics?

A single metric isn't sufficient because:

❌ High MRR ≠ high answer quality
❌ High faithfulness ≠ relevant answer
❌ Perfect retrieval ≠ perfect generation

**Using all 5 metrics gives complete picture of RAG performance.**

---

## Evaluation Limitations

### Small Dataset
- Only 20 documents and queries
- Intentional for development
- Future: expand to 100+ queries

### No Human Evaluation
- LLM-based evaluation can vary
- Future: add human annotation layer

### Specialized Domain
- Corpus focuses on RAG/LLM concepts
- Results may not generalize to other domains
- Domain adaptation needed for production

### Missing Metrics
- context_recall (not implemented yet)
- context_relevancy (not implemented yet)
- Statistical significance tests (need larger dataset)

---

## Roadmap

Currently implemented:
- ✅ Semantic/document-level chunking
- ✅ Parent-child chunking
- ✅ Hit@1, Hit@3, MRR metrics
- ✅ Faithfulness, Answer Relevancy metrics
- ✅ Automated comparison reports

Planned improvements:
- [ ] Add context_recall metric
- [ ] Add context_relevancy metric
- [ ] Expand benchmark to 100+ queries
- [ ] Add human evaluation framework
- [ ] Test 5+ chunking strategies
- [ ] Compare embedding models
- [ ] Statistical significance testing
- [ ] Multi-language support

---

## References

- [RAGAS Documentation](https://docs.ragas.io/)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://huggingface.co/docs/hub/sentence-transformers)
- [Chunking Strategies Survey](https://arxiv.org/abs/2401.07559)
"""

# ==================== BUILD GRADIO INTERFACE ====================

with gr.Blocks(title="RAGBench", theme=gr.themes.Soft()) as demo:
    gr.Markdown("### ZeroGPU Status")

    gpu_status = gr.Button("Check GPU")

    gpu_output = gr.Textbox(label="Status")

    gpu_status.click(
    fn=gpu_check,
    inputs=None,
    outputs=gpu_output
)
    
    # Tabs
    with gr.Tabs():
        with gr.TabItem("📊 Overview"):
            gr.Markdown(tab_overview())
        
        with gr.TabItem("🧪 About"):
            gr.Markdown(tab_about())
        
        with gr.TabItem("📈 Results"):
            gr.Markdown(tab_results())
        
        with gr.TabItem("🚀 How to Use"):
            gr.Markdown(tab_how_to_use())
        
        with gr.TabItem("📚 Documentation"):
            gr.Markdown(tab_documentation())
    
    # Footer
    gr.Markdown(
        """
        ---
        ### 🔗 Quick Links
        
        - **GitHub:** [github.com/GulrezQayyum/ragbench](https://github.com/GulrezQayyum/ragbench)
        - **Test Queries Dataset:** [HF Hub](https://huggingface.co/datasets/Gul55555/ragbench-queries?utm_source=chatgpt.com)
        - **Corpus Dataset:** [HF Hub](https://huggingface.co/datasets/Gul55555/ragbench-corpus?utm_source=chatgpt.com)
        - **Author:** [Gulrez Qayyum](https://github.com/GulrezQayyum)
        
        ### 📖 Built With
        ChromaDB • Sentence-Transformers • RAGAS • Groq API • Gradio
        
        ### 🤝 Contributing
        Found a bug? Have an idea? [Open an issue!](https://github.com/GulrezQayyum/ragbench)
        """
    )

# Launch
if __name__ == "__main__":
    demo.launch()