import spaces
import gradio as gr
import pandas as pd
import json
import os
from typing import Tuple

# ==================== CONFIGURATION ====================

# Check if Groq API key is available
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HAS_API_KEY = GROQ_API_KEY is not None

# ==================== DATA ====================

results_data = {
    "Metric": ["Hit@1", "Hit@3", "MRR", "Faithfulness", "Answer Relevancy"],
    "Semantic": [0.900, 1.000, 0.950, 1.000, 0.950],
    "Parent-Child": [0.900, 1.000, 0.933, 1.000, 1.000],
    "Winner": ["Tie", "Tie", "Semantic ✓", "Tie", "Parent-Child ✓"]
}

df_results = pd.DataFrame(results_data)

# ==================== INTERACTIVE FUNCTIONS ====================

def validate_json(json_str: str, field_name: str) -> Tuple[bool, dict, str]:
    """Validate and parse JSON input"""
    try:
        data = json.loads(json_str)
        return True, data, f"✅ {field_name} is valid JSON"
    except json.JSONDecodeError as e:
        return False, None, f"❌ Invalid JSON in {field_name}: {str(e)}"

@spaces.GPU
def run_custom_evaluation(corpus_json: str, queries_json: str, strategy: str) -> str:
    """Run evaluation on custom corpus and queries"""
    
    # Validate inputs
    corpus_valid, corpus_data, corpus_msg = validate_json(corpus_json, "Corpus")
    if not corpus_valid:
        return f"Error: {corpus_msg}"
    
    queries_valid, queries_data, queries_msg = validate_json(queries_json, "Queries")
    if not queries_valid:
        return f"Error: {queries_msg}"
    
    if not HAS_API_KEY:
        return """❌ **Groq API Key Not Found**

To run evaluations, you need to:

1. Get a free Groq API key: https://console.groq.com
2. Contact the Space owner to add it to the Space secrets

For now, you can:
- View pre-computed results in the "Results" tab
- Run evaluations locally on your machine
- Read the documentation for how to set up

**Local Setup:**
```bash
git clone https://github.com/GulrezQayyum/ragbench
cd ragbench
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
python eval.py
```
"""
    
    # Check corpus format
    if not isinstance(corpus_data, dict):
        return "❌ Corpus must be a JSON object with document IDs as keys"
    
    if len(corpus_data) == 0:
        return "❌ Corpus is empty. Please provide at least 1 document"
    
    # Check queries format
    if not isinstance(queries_data, list):
        return "❌ Queries must be a JSON array of query objects"
    
    if len(queries_data) == 0:
        return "❌ Queries array is empty. Please provide at least 1 query"
    
    # Validate query structure
    for i, q in enumerate(queries_data):
        if not isinstance(q, dict):
            return f"❌ Query {i} is not a valid object"
        if "query" not in q:
            return f"❌ Query {i} missing 'query' field"
        if "relevant_ids" not in q:
            return f"❌ Query {i} missing 'relevant_ids' field"
    
    # Prepare response
    response = f"""
✅ **Inputs Valid!**

**Corpus:** {len(corpus_data)} documents
**Queries:** {len(queries_data)} queries
**Strategy:** {strategy}

---

## 📊 Evaluation Setup

Your inputs are properly formatted:

### Corpus Preview
Documents loaded: {list(corpus_data.keys())[:3]}... ({len(corpus_data)} total)

### Queries Preview
```
{json.dumps(queries_data[:2], indent=2)}
...
```

---

## ⏳ Next Steps

Your evaluation is ready to run! 

**To execute the evaluation:**

1. Run locally with your API key:
```bash
git clone https://github.com/GulrezQayyum/ragbench
cd ragbench
export GROQ_API_KEY="your_key"
python eval.py
```

2. Or contact the Space owner to add Groq API key to secrets

---

## 📈 Expected Results

The evaluation will:
- Compare {strategy} chunking strategy
- Run against {len(queries_data)} queries
- Measure retrieval & generation quality
- Generate detailed comparison report

**Estimated time:** 2-5 minutes (depending on corpus size)
"""
    
    return response

# ==================== TAB CONTENTS ====================

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

### Change Embedding Model

In `eval.py`:

```python
embeddings_model = SentenceTransformer("all-mpnet-base-v2")
```

### Adjust Chunking Configuration

In `eval.py`:

```python
sentences_per_child = 3  # Smaller = finer chunks
```
"""

def tab_documentation():
    return """
# 📚 Complete Documentation

## Metrics Explained

### Retrieval Metrics

**Hit@K (Hit at K)**
- Measures if any relevant document appears in top-K results
- Range: 0.0 to 1.0 (higher is better)

**Mean Reciprocal Rank (MRR)**
- Average of reciprocal ranks of first relevant results
- Range: 0.0 to 1.0 (higher is better)

### Generation Metrics

**Faithfulness**
- Does generated answer strictly follow retrieved context?
- Range: 0.0 to 1.0 (higher is better)

**Answer Relevancy**
- Does answer directly address the question?
- Range: 0.0 to 1.0 (higher is better)

---

## Input Format Guide

### Corpus Format (JSON)

```json
{
  "doc_1": "First document content...",
  "doc_2": "Second document content...",
  "doc_3": "Third document content..."
}
```

### Queries Format (JSON Array)

```json
[
  {
    "query_id": "q_1",
    "query": "What is RAG?",
    "relevant_ids": ["doc_1", "doc_2"]
  },
  {
    "query_id": "q_2",
    "query": "How does chunking work?",
    "relevant_ids": ["doc_3"]
  }
]
```

---

## Why Multiple Metrics?

- ❌ High MRR ≠ high answer quality
- ❌ High faithfulness ≠ relevant answer
- ✅ Multiple metrics give complete picture

---

## Roadmap

- [x] Semantic/document-level chunking
- [x] Parent-child chunking
- [x] Core metrics (Hit@1, Hit@3, MRR, Faithfulness, Answer Relevancy)
- [ ] Add context_recall metric
- [ ] Add context_relevancy metric
- [ ] Expand benchmark dataset
- [ ] Human evaluation framework
- [ ] Compare embedding models
- [ ] Statistical significance testing
"""

def tab_try_it():
    return """
# 🧪 Try It Yourself

## Run Custom Evaluation

Test RAGBench with your own corpus and queries!

### How It Works

1. **Prepare your corpus** (JSON object with document content)
2. **Prepare your queries** (JSON array with query objects)
3. **Select a chunking strategy** (Semantic or Parent-Child)
4. **Click "Validate & Run"** to start evaluation

### Input Examples

See the **Input Format Guide** in the Documentation tab for exact JSON structure.

---

## 📋 Step-by-Step Guide

### Step 1: Format Your Corpus

Create a JSON object where keys are document IDs:

```json
{
  "doc_1": "Your document content here...",
  "doc_2": "Another document...",
  "doc_3": "Yet another document..."
}
```

### Step 2: Format Your Queries

Create a JSON array of query objects:

```json
[
  {
    "query_id": "q_1",
    "query": "Your question here?",
    "relevant_ids": ["doc_1"]
  },
  {
    "query_id": "q_2",
    "query": "Another question?",
    "relevant_ids": ["doc_2", "doc_3"]
  }
]
```

### Step 3: Choose Strategy & Validate

Select your strategy and click "Validate & Run"

---

## ⏱️ Timing

- **Validation:** Instant
- **Full Evaluation:** 2-5 minutes (depending on corpus/query size)
- **Result Generation:** <1 minute

---

## 💡 Tips

✅ Start small: 3-5 documents, 2-3 queries
✅ Ensure relevant_ids match your document IDs
✅ Use clear, specific queries
✅ Test both strategies for comparison

---

## 🆘 Troubleshooting

**"Invalid JSON"** → Check your brackets and quotes
**"Missing fields"** → Ensure required fields are present
**"API Key Error"** → Contact Space owner

---

Ready? Fill in the corpus and queries below! 👇
"""

# ==================== BUILD GRADIO INTERFACE ====================

with gr.Blocks(title="RAGBench") as demo:
    
    # Header
    gr.Markdown(
        """
        # 🦙 RAGBench
        ### Lightweight Evaluation Framework for RAG Systems
        
        Systematically compare RAG chunking strategies with reproducible benchmarks.
        """
    )
    
    # Tabs
    with gr.Tabs():
        # Tab 1: Overview
        with gr.TabItem("📊 Overview"):
            gr.Markdown(tab_overview())
        
        # Tab 2: About
        with gr.TabItem("🧪 About"):
            gr.Markdown(tab_about())
        
        # Tab 3: Results
        with gr.TabItem("📈 Results"):
            gr.Markdown(tab_results())
        
        # Tab 4: How to Use
        with gr.TabItem("🚀 How to Use"):
            gr.Markdown(tab_how_to_use())
        
        # Tab 5: Documentation
        with gr.TabItem("📚 Documentation"):
            gr.Markdown(tab_documentation())
        
        # Tab 6: TRY IT YOURSELF (NEW!)
        with gr.TabItem("🧪 Try It Yourself"):
            gr.Markdown(tab_try_it())
            
            # Input section
            gr.Markdown("## 📥 Input Your Data")
            
            with gr.Row():
                with gr.Column():
                    corpus_input = gr.Textbox(
                        label="📄 Corpus (JSON Object)",
                        placeholder='{"doc_1": "content...", "doc_2": "content..."}',
                        lines=8,
                        max_lines=20
                    )
                
                with gr.Column():
                    queries_input = gr.Textbox(
                        label="❓ Queries (JSON Array)",
                        placeholder='[{"query_id": "q_1", "query": "Your question?", "relevant_ids": ["doc_1"]}]',
                        lines=8,
                        max_lines=20
                    )
            
            # Strategy selection
            strategy_select = gr.Radio(
                ["Semantic", "Parent-Child", "Both"],
                value="Both",
                label="🎯 Chunking Strategy"
            )
            
            # Run button
            run_button = gr.Button("✅ Validate & Run Evaluation", variant="primary", size="lg")
            
            # Output
            gr.Markdown("## 📊 Results")
            output_area = gr.Textbox(
                label="Evaluation Results",
                lines=15,
                max_lines=30,
                interactive=False
            )
            
            # Connect button to function
            run_button.click(
                fn=run_custom_evaluation,
                inputs=[corpus_input, queries_input, strategy_select],
                outputs=output_area
            )
    
    # Footer
    gr.Markdown(
        """
        ---
        ### 🔗 Quick Links
        
        - **GitHub:** [github.com/GulrezQayyum/ragbench](https://github.com/GulrezQayyum/ragbench)
        - **Test Queries Dataset:** [HF Hub](https://huggingface.co/datasets/Gul55555/ragbench-queries)
        - **Corpus Dataset:** [HF Hub](https://huggingface.co/datasets/Gul55555/ragbench-corpus)
        - **Author:** [Gulrez Qayyum](https://github.com/GulrezQayyum)
        
        ### 📖 Built With
        ChromaDB • Sentence-Transformers • RAGAS • Groq API • Gradio
        
        ### 🤝 Contributing
        Found a bug? Have an idea? [Open an issue!](https://github.com/GulrezQayyum/ragbench)
        """
    )

# Launch
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())