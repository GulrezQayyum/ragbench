---
title: RAGBench
emoji: 🦙
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# RAGBench

**RAGBench** is a lightweight evaluation framework for benchmarking Retrieval-Augmented Generation (RAG) systems and comparing different chunking strategies.

The current benchmark compares:

* **Semantic / document-level chunking**
* **Parent-child chunking**

The framework evaluates both **retrieval quality** and **generated-answer quality** on a fixed test set.

---

## Overview

Chunking has a major impact on RAG performance.

A chunk that is too large may contain unnecessary information, while a chunk that is too small may lose the surrounding context needed to answer a question.

RAGBench provides a controlled way to compare chunking strategies using the same:

* Corpus
* Evaluation queries
* Embedding model
* Retrieval configuration
* Generation model
* Evaluation process

This makes it possible to measure how changes in chunking affect retrieval and answer quality.

---

## What RAGBench Evaluates

The current evaluation pipeline produces five metrics.

### Retrieval Metrics

#### Hit@1

Measures whether the first retrieved result is relevant to the query.

A score of `1.0` means the relevant result was ranked first for every evaluated query.

#### Hit@3

Measures whether a relevant result appears within the top three retrieved results.

A score of `1.0` means every query had a relevant result in the top three.

#### Mean Reciprocal Rank (MRR)

MRR measures how highly the first relevant result is ranked.

Higher MRR means relevant information tends to appear closer to the top of the retrieval results.

---

### Generation Metrics

#### Faithfulness

Measures whether the generated answer is supported by the retrieved context.

A high score indicates that the model is producing answers grounded in the retrieved information rather than introducing unsupported claims.

#### Answer Relevancy

Measures whether the generated answer directly addresses the user's question.

A high score indicates that the answer is focused and relevant to the query.

---

## Current Metrics

The current version of `eval.py` produces:

```text
Retrieval:
- hit_at_1
- hit_at_3
- mrr

Generation:
- faithfulness
- answer_relevancy
```

### Metrics Not Currently Produced

The current pipeline does **not** produce:

* `context_relevancy`
* `context_recall`

These metrics are therefore intentionally excluded from the current analysis report.

They may be added in a future version.

---

# Project Structure

```text
ragbench/
│
├── corpus.json
├── test_queries.json
├── eval.py
├── analyze_results.py
├── requirements.txt
├── README.md
│
└── results/
    ├── results_semantic.json
    ├── results_parent-child.json
    ├── query_analysis.json
    └── EVALUATION_REPORT.md
```

### Files

| File                                | Purpose                                                         |
| ----------------------------------- | --------------------------------------------------------------- |
| `corpus.json`                       | Evaluation corpus containing 20 RAG-related documents           |
| `test_queries.json`                 | 20 benchmark queries and their expected relevant document IDs   |
| `eval.py`                           | Runs retrieval, generation, and evaluation                      |
| `analyze_results.py`                | Compares the two strategies and generates the evaluation report |
| `requirements.txt`                  | Python dependencies                                             |
| `results/results_semantic.json`     | Raw Semantic strategy results                                   |
| `results/results_parent-child.json` | Raw Parent-Child strategy results                               |
| `results/query_analysis.json`       | Per-query comparison                                            |
| `results/EVALUATION_REPORT.md`      | Final benchmark report                                          |

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/GulrezQayyum/ragbench.git
cd ragbench
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses the following main dependencies:

* `groq` — LLM generation and evaluation
* `chromadb` — vector storage and similarity retrieval
* `sentence-transformers` — embedding generation
* `ragas` — RAG evaluation
* `datasets` — evaluation dataset handling
* `pandas` — result processing
* `python-dotenv` — environment variable management

---

# API Configuration

RAGBench uses Groq for LLM-based generation and evaluation.

Set your API key as an environment variable:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

Alternatively, create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit your API key to GitHub.

---

# Running the Benchmark

## Run Evaluation

Run:

```bash
python eval.py
```

The evaluation pipeline:

1. Loads the corpus.
2. Loads the benchmark queries.
3. Creates the Semantic/document-level representation.
4. Creates the Parent-Child representation.
5. Builds the retrieval index for each strategy.
6. Retrieves relevant context for each query.
7. Generates an answer.
8. Calculates retrieval metrics.
9. Calculates generation metrics.
10. Saves the raw evaluation results.

The current benchmark uses:

* **20 corpus documents**
* **20 evaluation queries**
* The same queries for both chunking strategies

Results are saved to:

```text
results/results_semantic.json
results/results_parent-child.json
```

---

# Analyze Results

After evaluation completes, run:

```bash
python analyze_results.py
```

This compares the two strategies and generates:

```text
results/EVALUATION_REPORT.md
results/query_analysis.json
```

The Markdown report contains:

* Executive summary
* Metric-level comparison
* Detailed metric statistics
* Query-level comparison
* Strategy winners
* Limitations
* Recommendations

---

# Chunking Strategies

## 1. Semantic / Document-Level Chunking

The current Semantic strategy treats each corpus document as a single retrieval unit.

In this benchmark, the corpus documents are already relatively focused on individual RAG concepts.

### Advantages

* Simple implementation
* Low chunk-management overhead
* Preserves the complete document context
* Works well when documents are already topically coherent

### Limitations

* A retrieved result may contain more information than the query requires
* Larger retrieval units can introduce additional context
* Less suitable for very long documents
* Does not provide fine-grained chunk retrieval

> **Important:** In the current implementation, "semantic chunking" is a document-level baseline rather than a full semantic-boundary chunking algorithm that dynamically splits documents based on embedding similarity.

---

## 2. Parent-Child Chunking

Parent-Child chunking divides documents into smaller child chunks while maintaining the larger parent document as surrounding context.

The child chunks provide more precise retrieval units, while the parent provides broader context to the generation step.

### Advantages

* More precise matching
* Smaller retrieval units
* Preserves broader document context
* Potentially useful when questions require multiple related pieces of information

### Limitations

* Creates more retrieval units
* Requires additional indexing and retrieval logic
* Parent context can contain information that is not directly relevant
* May introduce additional retrieval overhead

---

# Evaluation Methodology

RAGBench evaluates both strategies using the same benchmark.

For every query:

```text
Query
  │
  ▼
Chunking Strategy
  │
  ▼
Vector Retrieval
  │
  ▼
Retrieved Context
  │
  ▼
LLM Generation
  │
  ├── Retrieval Evaluation
  │     ├── Hit@1
  │     ├── Hit@3
  │     └── MRR
  │
  └── Generation Evaluation
        ├── Faithfulness
        └── Answer Relevancy
```

Because both strategies are evaluated using the same queries and corpus, their results can be compared directly.

---

# Current Benchmark Results

The current benchmark contains:

* **20 documents**
* **20 evaluation queries**

The latest evaluation produced the following results:

| Metric           | Semantic | Parent-Child | Result       |
| ---------------- | -------: | -----------: | ------------ |
| Hit@1            |    0.900 |        0.900 | Tie          |
| Hit@3            |    1.000 |        1.000 | Tie          |
| MRR              |    0.950 |        0.933 | Semantic     |
| Faithfulness     |    1.000 |        1.000 | Tie          |
| Answer Relevancy |    0.950 |        1.000 | Parent-Child |

---

## Retrieval Results

### Hit@1

```text
Semantic:     0.900
Parent-Child: 0.900
```

Both strategies performed identically.

Neither strategy has an advantage in whether a relevant result appears at rank 1.

### Hit@3

```text
Semantic:     1.000
Parent-Child: 1.000
```

Both strategies achieved perfect Hit@3.

Every benchmark query had a relevant result within the top three retrieved results.

### MRR

```text
Semantic:     0.950
Parent-Child: 0.933
```

Semantic achieved a slightly higher MRR.

This indicates that, on this benchmark, the first relevant result tended to appear slightly higher in the ranking with the Semantic strategy.

---

# Generation Results

### Faithfulness

```text
Semantic:     1.000
Parent-Child: 1.000
```

Both strategies achieved perfect faithfulness on the current test set.

This means the generated answers were judged to be fully supported by their retrieved context.

### Answer Relevancy

```text
Semantic:     0.950
Parent-Child: 1.000
```

Parent-Child achieved a higher Answer Relevancy score.

The difference suggests that Parent-Child retrieval produced context that allowed the generator to produce slightly more directly relevant answers for some queries.

---

# Query-Level Results

The current generation comparison uses:

```text
Generation Score =
(Faithfulness + Answer Relevancy) / 2
```

Across the 20 benchmark queries:

```text
Semantic wins:       0/20  (0%)
Parent-Child wins:   4/20  (20%)
Ties:               16/20  (80%)
```

### Queries Favoring Parent-Child

The four queries where Parent-Child achieved a higher generation score were:

1. **How does RRF combine results from different rankers?**
2. **What is the benefit of asking a broader question before the specific one?**
3. **How does a system decide which retrieval approach to use for a given question?**
4. **How can questions requiring multiple connected facts be answered?**

For each of these queries:

```text
Parent-Child: 1.000
Semantic:     0.875
Margin:       0.125
```

No queries in the current benchmark favored Semantic on the combined generation score.

---

# Interpreting the Results

The benchmark does **not** show that one strategy universally outperforms the other.

Instead, the results show different strengths.

### Semantic / Document-Level

Semantic achieved:

* Equal Hit@1
* Equal Hit@3
* Higher MRR
* Equal Faithfulness
* Slightly lower Answer Relevancy

This suggests that the document-level strategy performed very well for ranking relevant information near the top.

### Parent-Child

Parent-Child achieved:

* Equal Hit@1
* Equal Hit@3
* Slightly lower MRR
* Equal Faithfulness
* Higher Answer Relevancy

This suggests that Parent-Child retrieval can provide useful contextual information for generating directly relevant answers, despite its slightly lower MRR on this benchmark.

---

# Is There an Overall Winner?

There is **no clear universal winner**.

The results are:

```text
                 Semantic    Parent-Child
Hit@1               0.900       0.900
Hit@3               1.000       1.000
MRR                 0.950       0.933
Faithfulness        1.000       1.000
Answer Relevancy    0.950       1.000
```

Semantic wins MRR.

Parent-Child wins Answer Relevancy.

The remaining metrics are tied.

Therefore, the appropriate conclusion is:

> **On this 20-query benchmark, Semantic/document-level chunking achieved slightly better retrieval ranking performance, while Parent-Child chunking achieved better Answer Relevancy. Neither strategy is a universal winner across all evaluated metrics.**

---

# Limitations

The current results should be interpreted within the scope of this benchmark.

### Small Evaluation Set

The benchmark currently contains only:

```text
20 documents
20 queries
```

This is useful for demonstrating the evaluation framework, but a larger dataset would provide stronger evidence.

### LLM-Based Evaluation

Faithfulness and Answer Relevancy depend on an LLM-based judge.

LLM evaluation can vary depending on:

* Judge model
* Prompting
* Model behavior
* API response variability

### No Human Evaluation

The current benchmark does not include human scoring.

Human evaluation would provide an additional validation layer for generated-answer quality.

### Specialized Corpus

The corpus focuses on RAG concepts.

Results from this dataset may not generalize to:

* Legal documents
* Financial documents
* Medical documents
* Technical documentation
* Customer-support knowledge bases
* Other production domains

### Limited Context-Level Evaluation

The current version does not produce:

```text
context_relevancy
context_recall
```

Therefore, the benchmark currently cannot provide direct metric-level comparisons for those dimensions.

---

# Why Use Multiple Metrics?

A single metric is not enough to evaluate a RAG system.

For example:

```text
High MRR
```

does not necessarily mean:

```text
High answer quality
```

Similarly:

```text
High Answer Relevancy
```

does not necessarily mean:

```text
The answer is grounded in the retrieved context
```

RAGBench therefore separates evaluation into two stages:

### Retrieval

```text
Hit@1
Hit@3
MRR
```

### Generation

```text
Faithfulness
Answer Relevancy
```

This makes it easier to identify whether a problem originates from retrieval or generation.

---

# Reproducibility

RAGBench uses a fixed benchmark:

```text
Corpus → test_queries.json → Retrieval → Generation → Evaluation
```

Both strategies are evaluated against the same queries.

This allows changes to the following components to be tested systematically:

* Chunking strategy
* Chunk size
* Parent-child configuration
* Retrieval `top-k`
* Embedding model
* Generation model
* Evaluation configuration

---

# Customizing the Benchmark

## Add Evaluation Queries

Add additional entries to:

```text
test_queries.json
```

Example:

```json
{
  "query_id": "q_21",
  "query": "Your evaluation question?",
  "relevant_ids": ["doc_05"]
}
```

Adding more queries is one of the most useful ways to improve the reliability of the benchmark.

---

## Change Retrieval Top-K

The retrieval configuration can be changed in `eval.py`.

For example:

```python
top_k=5
```

instead of:

```python
top_k=3
```

Changing `top-k` allows you to study how retrieval depth affects both retrieval and generation quality.

---

## Change Parent-Child Configuration

The Parent-Child strategy can be configured through its chunking function in `eval.py`.

For example:

```python
def chunk_corpus_parent_child(self, sentences_per_child=3):
    ...
```

Increasing the number of sentences per child creates larger child chunks.

Smaller children generally provide finer-grained retrieval, while larger children provide more local context.

---

## Change the Embedding Model

The embedding model can also be changed in `eval.py`.

For example:

```python
self.embeddings_model = SentenceTransformer(
    "all-mpnet-base-v2"
)
```

Different embedding models can produce different retrieval rankings, so changing the embedding model should be treated as a separate experiment.

---

# Troubleshooting

## `GROQ_API_KEY not set`

Set the API key before running the evaluation:

```bash
export GROQ_API_KEY="your_groq_api_key"
python eval.py
```

Or place the key in `.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## RAGAS Evaluation Errors

If evaluation fails, common causes include:

* API rate limits
* Request timeouts
* Invalid API credentials
* Temporary API failures
* Dependency/version incompatibilities

Check the terminal output for the specific failing request before changing the evaluation logic.

---

## Slow Evaluation

LLM-based generation and evaluation require API calls.

Runtime depends on:

* Number of queries
* Number of evaluation metrics
* API latency
* API rate limits
* Model response time

The current 20-query benchmark is intended to remain lightweight enough for development and experimentation.

---

# Future Improvements

The current implementation provides a foundation for expanding RAGBench.

Potential future improvements include:

### 1. Add Context Relevancy

Evaluate whether retrieved chunks are actually relevant to the query.

### 2. Add Context Recall

Measure whether the retrieved context contains the information needed to answer the question.

### 3. Expand the Dataset

Increase the number and diversity of:

* Documents
* Queries
* Query types
* Multi-hop questions
* Difficult retrieval cases

### 4. Add Human Evaluation

Compare automated evaluation against human judgments.

### 5. Test More Chunking Strategies

Potential strategies include:

* Fixed-size chunking
* Recursive chunking
* Sentence-based chunking
* Semantic boundary chunking
* Sliding-window chunking
* Hierarchical chunking

### 6. Test More Embedding Models

Compare retrieval performance across different embedding models.

### 7. Add Statistical Analysis

With a larger dataset, confidence intervals and statistical significance testing could be added to determine whether observed differences are meaningful.

---

# Project Workflow

The complete workflow is:

```text
                 ┌─────────────────┐
                 │    Corpus       │
                 │ 20 documents    │
                 └────────┬────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
      ┌───────────────┐       ┌────────────────┐
      │ Semantic /    │       │ Parent-Child   │
      │ Document-Level│       │ Chunking       │
      └───────┬───────┘       └───────┬────────┘
              │                       │
              ▼                       ▼
      ┌───────────────┐       ┌────────────────┐
      │   Retrieval   │       │   Retrieval    │
      └───────┬───────┘       └───────┬────────┘
              │                       │
              ▼                       ▼
      ┌───────────────┐       ┌────────────────┐
      │   Generation  │       │   Generation   │
      └───────┬───────┘       └───────┬────────┘
              │                       │
              └───────────┬───────────┘
                          ▼
                 ┌──────────────────┐
                 │    Evaluation    │
                 ├──────────────────┤
                 │ Hit@1            │
                 │ Hit@3            │
                 │ MRR              │
                 │ Faithfulness     │
                 │ Answer Relevancy │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Results Analysis │
                 └──────────────────┘
```

---

# Example Research Question

RAGBench can be used to investigate questions such as:

> **Does Parent-Child chunking improve RAG answer quality compared with document-level retrieval?**

The current benchmark suggests:

* Retrieval ranking is slightly better with the Semantic strategy according to MRR.
* Parent-Child achieves better Answer Relevancy.
* Hit@1 and Hit@3 are identical.
* Faithfulness is identical.
* Therefore, the answer depends on which aspect of RAG performance is most important.

---

# Portfolio Value

RAGBench demonstrates practical experience with:

* Retrieval-Augmented Generation
* Vector databases
* Embedding models
* Chunking strategies
* LLM-based evaluation
* RAGAS
* Retrieval metrics
* Generation metrics
* Benchmark design
* Experiment comparison
* Result analysis

The project is designed not simply to build a RAG pipeline, but to **measure and compare how design decisions affect RAG performance**.

---

# Current Result Summary

The latest benchmark result can be summarized as:

```text
                    Semantic    Parent-Child
                    --------    -----------
Hit@1                 0.900        0.900
Hit@3                 1.000        1.000
MRR                   0.950        0.933
Faithfulness          1.000        1.000
Answer Relevancy      0.950        1.000
```

### Conclusion

**Semantic/document-level chunking** achieved slightly better retrieval ranking according to MRR.

**Parent-Child chunking** achieved better Answer Relevancy and won on 4 of the 20 query-level generation comparisons.

However, **16 of 20 queries were ties**, and neither strategy dominated the benchmark across all metrics.

The current evidence therefore supports a **trade-off rather than a universal winner**.

---

# Evaluation Report

For the complete benchmark analysis, see:

```text
results/EVALUATION_REPORT.md
```

The report contains the detailed metric statistics and query-level comparison generated from the current evaluation results.

---

# License

This project is intended for educational, experimental, and research purposes.
