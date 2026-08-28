
import streamlit as st
import json
import pandas as pd
from pathlib import Path
import os

# Page config
st.set_page_config(
    page_title="RAGBench - RAG Evaluation Framework",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .title-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .winner {
        background: #d1fae5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="title-box">
    <h1>🦙 RAGBench</h1>
    <p>Lightweight Evaluation Framework for Retrieval-Augmented Generation Systems</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["📊 Overview", "🧪 About RAGBench", "📈 Current Results", "🚀 How to Use", "📚 Documentation"]
)

# ==================== PAGE 1: OVERVIEW ====================
if page == "📊 Overview":
    st.header("What is RAGBench?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        RAGBench is a **controlled evaluation framework** for comparing different RAG chunking strategies:
        
        - **Semantic/Document-Level Chunking**
        - **Parent-Child Chunking**
        
        Using the same:
        - ✓ Corpus (documents)
        - ✓ Evaluation queries
        - ✓ Embedding model
        - ✓ LLM for generation
        - ✓ Evaluation metrics
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Metrics Evaluated
        
        **Retrieval:**
        - Hit@1, Hit@3
        - Mean Reciprocal Rank (MRR)
        
        **Generation:**
        - Faithfulness
        - Answer Relevancy
        
        Separate evaluation helps identify if problems come from retrieval or generation.
        """)
    
    st.divider()
    
    st.subheader("Why Chunking Matters")
    st.markdown("""
    Chunks that are **too large** contain noise.  
    Chunks that are **too small** lose context.  
    
    **RAGBench measures this trade-off scientifically.**
    """)

# ==================== PAGE 2: ABOUT ====================
elif page == "🧪 About RAGBench":
    st.header("The Problem RAGBench Solves")
    
    st.markdown("""
    ### ❓ The Chunking Strategy Question
    
    When building RAG systems, developers ask:
    > "Should I use semantic chunking or parent-child chunking?"
    
    **Before RAGBench:** Anecdotal evidence, guessing, trial-and-error
    
    **With RAGBench:** Systematic, reproducible benchmark with clear metrics
    
    ---
    
    ### 🏗️ Project Structure
    """)
    
    st.code("""
ragbench/
├── corpus.json              # 20 evaluation documents
├── test_queries.json        # 20 benchmark queries
├── eval.py                  # Runs retrieval + generation
├── analyze_results.py       # Compares strategies
├── requirements.txt         # Dependencies
└── results/
    ├── results_semantic.json
    ├── results_parent-child.json
    └── EVALUATION_REPORT.md
    """, language="bash")
    
    st.divider()
    
    st.subheader("Key Technologies")
    tech_cols = st.columns(4)
    with tech_cols[0]:
        st.metric("Vector Store", "ChromaDB")
    with tech_cols[1]:
        st.metric("Embeddings", "Sentence-Transformers")
    with tech_cols[2]:
        st.metric("Evaluation", "RAGAS")
    with tech_cols[3]:
        st.metric("LLM", "Groq API")

# ==================== PAGE 3: RESULTS ====================
elif page == "📈 Current Results":
    st.header("Benchmark Results: Semantic vs Parent-Child")
    
    # Results data
    results_data = {
        "Metric": ["Hit@1", "Hit@3", "MRR", "Faithfulness", "Answer Relevancy"],
        "Semantic": [0.900, 1.000, 0.950, 1.000, 0.950],
        "Parent-Child": [0.900, 1.000, 0.933, 1.000, 1.000],
        "Winner": ["Tie", "Tie", "Semantic ✓", "Tie", "Parent-Child ✓"]
    }
    
    df = pd.DataFrame(results_data)
    
    st.subheader("Retrieval + Generation Metrics")
    st.dataframe(df, use_container_width=True)
    
    st.divider()
    
    # Detailed analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🟦 Semantic/Document-Level
        
        **Strengths:**
        - Higher MRR (0.950)
        - Better ranking of relevant results
        - Simpler implementation
        
        **Trade-offs:**
        - Slightly lower Answer Relevancy
        - May include extra context
        """)
    
    with col2:
        st.markdown("""
        ### 🟪 Parent-Child Chunking
        
        **Strengths:**
        - Perfect Answer Relevancy (1.000)
        - More focused context
        - Better for multi-fact questions
        
        **Trade-offs:**
        - Slightly lower MRR (0.933)
        - More retrieval overhead
        """)
    
    st.divider()
    
    st.subheader("Query-Level Analysis")
    st.markdown("""
    Across 20 benchmark queries:
    
    | Outcome | Count |
    |---------|-------|
    | Semantic wins | 0/20 (0%) |
    | Parent-Child wins | 4/20 (20%) |
    | Ties | 16/20 (80%) |
    
    **Conclusion:** No universal winner. Trade-off between retrieval ranking and generation quality.
    """)

# ==================== PAGE 4: HOW TO USE ====================
elif page == "🚀 How to Use":
    st.header("Running RAGBench Locally")
    
    st.subheader("Step 1: Clone & Install")
    st.code("""
git clone https://github.com/GulrezQayyum/ragbench.git
cd ragbench
pip install -r requirements.txt
    """, language="bash")
    
    st.subheader("Step 2: Set Up Groq API")
    st.code("""
# Create .env file
export GROQ_API_KEY="your_api_key_here"

# Or create .env file in project root
echo "GROQ_API_KEY=your_api_key_here" > .env
    """, language="bash")
    
    st.info("Get free API key at: https://console.groq.com")
    
    st.subheader("Step 3: Run Evaluation")
    st.code("""
python eval.py
    """, language="bash")
    
    st.success("Evaluation results saved to `results/` folder")
    
    st.subheader("Step 4: Analyze Results")
    st.code("""
python analyze_results.py
    """, language="bash")
    
    st.markdown("Check `results/EVALUATION_REPORT.md` for full analysis")
    
    st.divider()
    
    st.subheader("Customization Options")
    st.markdown("""
    **Add More Queries:**
    Edit `test_queries.json` and add new entries
    
    **Change Embedding Model:**
    In `eval.py`, modify the embedding model
    
    **Adjust Chunking Strategy:**
    Modify `chunk_corpus_parent_child()` in `eval.py`
    
    **Change Top-K Retrieval:**
    Adjust `top_k` parameter in retrieval config
    """)

# ==================== PAGE 5: DOCUMENTATION ====================
elif page == "📚 Documentation":
    st.header("Complete Documentation")
    
    tabs = st.tabs(["Quick Start", "Metrics Explained", "Architecture", "Limitations", "Future Work"])
    
    with tabs[0]:
        st.markdown("""
        ### Quick Start (5 minutes)
        
        1. Clone: `git clone https://github.com/GulrezQayyum/ragbench.git`
        2. Install: `pip install -r requirements.txt`
        3. Add API key: `export GROQ_API_KEY=...`
        4. Run: `python eval.py`
        5. Analyze: `python analyze_results.py`
        
        Results appear in `results/EVALUATION_REPORT.md`
        """)
    
    with tabs[1]:
        st.markdown("""
        ### Retrieval Metrics
        
        **Hit@1:** Is the most relevant result ranked first?
        
        **Hit@3:** Is a relevant result in top 3?
        
        **MRR:** How high does the first relevant result rank?
        
        ### Generation Metrics
        
        **Faithfulness:** Is the answer grounded in retrieved context?
        
        **Answer Relevancy:** Does the answer address the query?
        """)
    
    with tabs[2]:
        st.markdown("""
        ### System Architecture
        
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
        Metrics (Hit@1, MRR, Faithfulness, etc.)
        ```
        """)
    
    with tabs[3]:
        st.markdown("""
        ### Current Limitations
        
        - Small benchmark (20 queries, 20 documents)
        - No human evaluation
        - Specialized to RAG domain
        - Limited to English
        - No context_recall or context_relevancy metrics
        
        These are intentional for now—they make the framework lightweight.
        """)
    
    with tabs[4]:
        st.markdown("""
        ### Roadmap
        
        - [ ] Add Context Relevancy
        - [ ] Add Context Recall
        - [ ] Expand benchmark dataset
        - [ ] Add human evaluation
        - [ ] Test more chunking strategies
        - [ ] Compare embedding models
        - [ ] Statistical significance testing
        """)

st.divider()

# Footer
st.markdown("""
---
### 🔗 Links

**Repository:** [github.com/GulrezQayyum/ragbench](https://github.com/GulrezQayyum/ragbench)

**Author:** [Gulrez Qayyum](https://github.com/GulrezQayyum)

---

### 🤝 Contributing

Have ideas? Found a bug? Want to collaborate?
Open an issue or PR on the [GitHub repository](https://github.com/GulrezQayyum/ragbench)!

### 📖 References

- [RAGAS Documentation](https://docs.ragas.io/)
- [ChromaDB Vector Store](https://www.trychroma.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
""")