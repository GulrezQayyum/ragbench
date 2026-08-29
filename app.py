import os
import json
from typing import Tuple

import gradio as gr
import pandas as pd
import spaces

from eval import RAGEvaluator, ChunkingConfig


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HAS_API_KEY = bool(GROQ_API_KEY)

print("🔑 Groq API key detected:", HAS_API_KEY)


# ============================================================
# PRE-COMPUTED RESULTS
# ============================================================

results_data = {
    "Metric": [
        "Hit@1",
        "Hit@3",
        "MRR",
        "Faithfulness",
        "Answer Relevancy",
    ],
    "Semantic": [
        0.900,
        1.000,
        0.950,
        1.000,
        0.950,
    ],
    "Parent-Child": [
        0.900,
        1.000,
        0.933,
        1.000,
        1.000,
    ],
    "Winner": [
        "Tie",
        "Tie",
        "Semantic ✓",
        "Tie",
        "Parent-Child ✓",
    ],
}

df_results = pd.DataFrame(results_data)


# ============================================================
# JSON VALIDATION
# ============================================================

def validate_json(
    json_str: str,
    field_name: str,
) -> Tuple[bool, object, str]:
    """Validate and parse JSON input."""

    try:
        data = json.loads(json_str)

        return (
            True,
            data,
            f"✅ {field_name} is valid JSON",
        )

    except json.JSONDecodeError as e:

        return (
            False,
            None,
            f"❌ Invalid JSON in {field_name}: {str(e)}",
        )


# ============================================================
# CUSTOM EVALUATION
# ============================================================

@spaces.GPU
def run_custom_evaluation(
    corpus_json: str,
    queries_json: str,
    strategy: str,
) -> str:
    """Run evaluation on custom corpus and queries."""

    # --------------------------------------------------------
    # Validate Corpus JSON
    # --------------------------------------------------------

    corpus_valid, corpus_data, corpus_msg = validate_json(
        corpus_json,
        "Corpus",
    )

    if not corpus_valid:
        return corpus_msg

    # --------------------------------------------------------
    # Validate Queries JSON
    # --------------------------------------------------------

    queries_valid, queries_data, queries_msg = validate_json(
        queries_json,
        "Queries",
    )

    if not queries_valid:
        return queries_msg

    # --------------------------------------------------------
    # Check API Key
    # --------------------------------------------------------

    if not HAS_API_KEY:
        return """
# ❌ Groq API Key Not Found

The Space needs a Groq API key to run the LLM evaluation.

Please add `GROQ_API_KEY` to your Hugging Face Space Secrets.

The pre-computed benchmark results are still available
in the **Results** tab.
"""

    # --------------------------------------------------------
    # Validate Strategy
    # --------------------------------------------------------

    if strategy == "Both":
        return """
# ⚠️ Both Strategies Selected

For this version, please run one strategy at a time.

Select either:

- **Semantic**
- **Parent-Child**
"""

    # --------------------------------------------------------
    # Validate Corpus
    # --------------------------------------------------------

    if not isinstance(corpus_data, list):
        return "❌ Corpus must be a JSON array of document objects."

    if len(corpus_data) == 0:
        return "❌ Corpus is empty."

    for i, doc in enumerate(corpus_data):

        if not isinstance(doc, dict):
            return f"❌ Document {i} is not a valid object."

        if "doc_id" not in doc:
            return f"❌ Document {i} is missing 'doc_id'."

        if "content" not in doc:
            return f"❌ Document {i} is missing 'content'."

        if not isinstance(doc["doc_id"], str):
            return f"❌ Document {i} has invalid 'doc_id'."

        if not isinstance(doc["content"], str):
            return f"❌ Document {i} has invalid 'content'."

    # --------------------------------------------------------
    # Validate Queries
    # --------------------------------------------------------

    if not isinstance(queries_data, list):
        return "❌ Queries must be a JSON array."

    if len(queries_data) == 0:
        return "❌ Queries array is empty."

    for i, query in enumerate(queries_data):

        if not isinstance(query, dict):
            return f"❌ Query {i} is not a valid object."

        if "query_id" not in query:
            return f"❌ Query {i} is missing 'query_id'."

        if "query" not in query:
            return f"❌ Query {i} is missing 'query'."

        if "relevant_ids" not in query:
            return f"❌ Query {i} is missing 'relevant_ids'."

    # --------------------------------------------------------
    # Save Temporary Input Files
    # --------------------------------------------------------

    temp_corpus = "custom_corpus.json"
    temp_queries = "custom_queries.json"

    try:

        with open(
            temp_corpus,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                corpus_data,
                f,
                indent=2,
            )

        with open(
            temp_queries,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                queries_data,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # Create Evaluator
        # ----------------------------------------------------

        evaluator = RAGEvaluator(
            corpus_path=temp_corpus,
            queries_path=temp_queries,
        )

        # ----------------------------------------------------
        # Create Chunking Configuration
        # ----------------------------------------------------

        if strategy == "Semantic":

            config = ChunkingConfig(
                name="custom_semantic",
                strategy="semantic",
                chunk_size=512,
                overlap=0,
            )

        else:

            config = ChunkingConfig(
                name="custom_parent_child",
                strategy="parent-child",
                chunk_size=256,
                overlap=50,
            )

        # ----------------------------------------------------
        # Run Evaluation
        # ----------------------------------------------------

        results = evaluator.run_evaluation(
            config=config,
            top_k=3,
            resume=False,
        )

        # ----------------------------------------------------
        # Calculate Summary
        # ----------------------------------------------------

        metrics = [
            "hit_at_1",
            "hit_at_3",
            "mrr",
            "faithfulness",
            "answer_relevancy",
        ]

        summary = {}

        for metric in metrics:

            scores = [
                float(result[metric])
                for result in results
                if result.get(metric) is not None
            ]

            if scores:
                summary[metric] = sum(scores) / len(scores)
            else:
                summary[metric] = None

        # ----------------------------------------------------
        # Format Scores
        # ----------------------------------------------------

        def format_score(value):

            if value is None:
                return "N/A"

            return f"{value:.3f}"

        # ----------------------------------------------------
        # Format Final Output
        # ----------------------------------------------------

        output = f"""
# ✅ Evaluation Complete

## Configuration

- **Strategy:** {strategy}
- **Documents:** {len(corpus_data)}
- **Queries:** {len(queries_data)}
- **Top-K:** 3

---

## 📊 Results

| Metric | Score |
|---|---:|
| Hit@1 | {format_score(summary["hit_at_1"])} |
| Hit@3 | {format_score(summary["hit_at_3"])} |
| MRR | {format_score(summary["mrr"])} |
| Faithfulness | {format_score(summary["faithfulness"])} |
| Answer Relevancy | {format_score(summary["answer_relevancy"])} |

---

## 🔍 Interpretation

**Hit@1:** Whether a relevant document was retrieved first.

**Hit@3:** Whether a relevant document appeared within the top three results.

**MRR:** How highly the first relevant result was ranked.

**Faithfulness:** Whether the generated answer is supported by the retrieved context.

**Answer Relevancy:** Whether the generated answer directly addresses the question.

---

✅ Evaluation finished successfully.
"""

        return output

    except Exception as e:

        return f"""
# ❌ Evaluation Failed

**Error:**

```text
{str(e)}
```

Please check the Space logs for the complete traceback.
"""


# ============================================================
# TAB CONTENT
# ============================================================

def tab_overview():
    return """
# 🦙 RAGBench - Overview

## What is RAGBench?

RAGBench is a **controlled evaluation framework** for comparing different RAG chunking strategies.

### Strategies

* **Semantic / Document-Level Chunking**
* **Parent-Child Chunking**

Both strategies use the same corpus, queries, embedding model, and evaluation metrics.

## 🎯 Purpose

When building RAG systems, one important question is:

> What is the best way to chunk documents?

RAGBench provides a data-driven way to investigate this question.

## 📊 Metrics

### Retrieval Quality

* **Hit@1** — Is a relevant result ranked first?
* **Hit@3** — Is a relevant result in the top three?
* **MRR** — How highly does the first relevant result rank?

### Generation Quality

* **Faithfulness** — Is the answer grounded in the context?
* **Answer Relevancy** — Does the answer address the query?

---

Separating retrieval and generation metrics helps identify where problems originate.
"""


def tab_about():
    return """
# 🧪 About RAGBench

## The Problem

RAG developers often choose chunking strategies using:

* ❌ Gut feeling
* ❌ Blog posts
* ❌ Trial and error
* ❌ Random experimentation

## The Solution

RAGBench provides:

* ✅ Fixed corpus
* ✅ Fixed benchmark queries
* ✅ Multiple evaluation metrics
* ✅ Reproducible experiments
* ✅ Customizable evaluation

## 🏗️ How It Works

```text
Query
  ↓
Chunking Strategy
  ↓
Vector Retrieval
  ↓
Retrieved Context
  ↓
LLM Generation
  ↓
LLM-as-Judge Evaluation
  ↓
Metrics
```

## 🔧 Technologies

* ChromaDB
* Sentence-Transformers
* Groq API
* Python
* Gradio
"""


def tab_results():
    results_html = df_results.to_html(index=False)

    return f"""
# 📈 Current Benchmark Results

## Semantic vs Parent-Child Comparison

{results_html}

## Key Findings

### Retrieval Performance

**Hit@1:** 0.900 for both strategies.

**Hit@3:** 1.000 for both strategies.

**MRR:**

* Semantic: 0.950
* Parent-Child: 0.933

Semantic performs slightly better on ranking.

### Generation Performance

**Faithfulness:**

* Semantic: 1.000
* Parent-Child: 1.000

**Answer Relevancy:**

* Semantic: 0.950
* Parent-Child: 1.000

Parent-Child performs better on answer relevancy.

## Query-Level Analysis

| Outcome           | Count | Percentage |
| ----------------- | ----: | ---------: |
| Semantic wins     |  0/20 |         0% |
| Parent-Child wins |  4/20 |        20% |
| Ties              | 16/20 |        80% |

## Interpretation

There is no universal winner.

The better strategy depends on the metric and application requirements.
"""


def tab_how_to_use():
    return """
# 🚀 How to Use RAGBench

## Quick Start

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

Get a Groq API key from:

https://console.groq.com

Then set:

```bash
export GROQ_API_KEY="your_api_key_here"
```

### 4. Run Evaluation

```bash
python eval.py
```

### 5. Analyze Results

```bash
python analyze_results.py
```

Results are stored in the `results/` directory.
"""


def tab_documentation():
    return """
# 📚 Complete Documentation

## Retrieval Metrics

### Hit@K

Measures whether a relevant document appears within the top-K retrieved results.

Range: 0 to 1.

### MRR

Mean Reciprocal Rank measures the ranking position of the first relevant result.

Range: 0 to 1.

## Generation Metrics

### Faithfulness

Measures whether generated claims are supported by the retrieved context.

### Answer Relevancy

Measures whether the generated answer addresses the query.

## Why Multiple Metrics?

A system can have strong retrieval but weak generation.

It can also generate relevant answers from poor retrieval.

Therefore, multiple metrics provide a more complete evaluation.

---

## Roadmap

* [x] Semantic/document-level chunking
* [x] Parent-child chunking
* [x] Hit@1
* [x] Hit@3
* [x] MRR
* [x] Faithfulness
* [x] Answer Relevancy
* [ ] Add context recall
* [ ] Add context relevancy
* [ ] Expand benchmark dataset
* [ ] Human evaluation framework
* [ ] Compare embedding models
* [ ] Statistical significance testing
"""


def tab_try_it():
    return """
# 🧪 Try It Yourself

Test RAGBench with your own corpus and benchmark queries.

## Corpus Format

```json
[
  {
    "doc_id": "doc_01",
    "content": "Your document content..."
  },
  {
    "doc_id": "doc_02",
    "content": "Another document..."
  }
]
```

## Query Format

```json
[
  {
    "query_id": "q_01",
    "query": "What is RAG?",
    "relevant_ids": ["doc_01"]
  }
]
```

## Steps

1. Enter your corpus.
2. Enter your queries.
3. Select a chunking strategy.
4. Click **Validate & Run Evaluation**.

For a first experiment, start with a small corpus and a few queries.
"""


# ============================================================
# BUILD GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="RAGBench") as demo:

    # Header
    gr.Markdown(
        """
# 🦙 RAGBench

### Lightweight Evaluation Framework for RAG Systems

Systematically compare RAG chunking strategies using reproducible benchmarks.
"""
    )

    # Tabs
    with gr.Tabs():

        # Overview
        with gr.TabItem("📊 Overview"):
            gr.Markdown(tab_overview())

        # About
        with gr.TabItem("🧪 About"):
            gr.Markdown(tab_about())

        # Results
        with gr.TabItem("📈 Results"):
            gr.Markdown(tab_results())

        # How to Use
        with gr.TabItem("🚀 How to Use"):
            gr.Markdown(tab_how_to_use())

        # Documentation
        with gr.TabItem("📚 Documentation"):
            gr.Markdown(tab_documentation())

        # Try It Yourself
        with gr.TabItem("🧪 Try It Yourself"):
            gr.Markdown(tab_try_it())

            gr.Markdown("## 📥 Input Your Data")

            with gr.Row():

                with gr.Column():
                    corpus_input = gr.Textbox(
                        label="📄 Corpus (JSON Array)",
                        placeholder=(
                            '[{"doc_id": "doc_01", '
                            '"content": "Document content..."}]'
                        ),
                        lines=8,
                        max_lines=20,
                    )

                with gr.Column():
                    queries_input = gr.Textbox(
                        label="❓ Queries (JSON Array)",
                        placeholder=(
                            '[{"query_id": "q_01", '
                            '"query": "Your question?", '
                            '"relevant_ids": ["doc_01"]}]'
                        ),
                        lines=8,
                        max_lines=20,
                    )

            strategy_select = gr.Radio(
                choices=[
                    "Semantic",
                    "Parent-Child",
                    "Both",
                ],
                value="Both",
                label="🎯 Chunking Strategy",
            )

            run_button = gr.Button(
                "✅ Validate & Run Evaluation",
                variant="primary",
                size="lg",
            )

            gr.Markdown("## 📊 Results")

            output_area = gr.Textbox(
                label="Evaluation Results",
                lines=15,
                max_lines=30,
                interactive=False,
            )

            run_button.click(
                fn=run_custom_evaluation,
                inputs=[
                    corpus_input,
                    queries_input,
                    strategy_select,
                ],
                outputs=output_area,
            )

    # Footer
    gr.Markdown(
        """
---

### 🔗 Quick Links

* **GitHub:** https://github.com/GulrezQayyum/ragbench
* **Test Queries Dataset:** https://huggingface.co/datasets/Gul55555/ragbench-queries
* **Corpus Dataset:** https://huggingface.co/datasets/Gul55555/ragbench-corpus

### 📖 Built With

ChromaDB • Sentence-Transformers • RAGAS • Groq API • Gradio

### 🤝 Contributing

Found a bug or have an idea? Open an issue on GitHub.
"""
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft()
    )