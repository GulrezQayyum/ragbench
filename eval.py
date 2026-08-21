#!/usr/bin/env python3

"""
RAGBench Evaluation Runner

Compares semantic chunking vs parent-child chunking
using Groq LLM-based evaluation metrics.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq


@dataclass
class ChunkingConfig:
    name: str
    strategy: str
    chunk_size: int
    overlap: int


class RAGEvaluator:

    def __init__(
        self,
        corpus_path: str = "corpus.json",
        queries_path: str = "test_queries.json",
    ):
        """Initialize evaluator."""

        self.corpus = self._load_json(corpus_path)
        self.queries = self._load_json(queries_path)

        self.groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.embeddings_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.chroma_client = chromadb.Client()

        print(f"✅ Loaded {len(self.corpus)} documents")
        print(f"✅ Loaded {len(self.queries)} test queries")

    # ============================================================
    # DATA LOADING
    # ============================================================

    @staticmethod
    def _load_json(path: str) -> Any:
        """Load JSON file."""

        with open(path, "r") as f:
            return json.load(f)

    # ============================================================
    # CHUNKING
    # ============================================================

    def chunk_corpus_semantic(self) -> Dict[str, Dict[str, Any]]:
        """
        Semantic chunking.

        For this benchmark, each document is kept as one chunk.
        """

        chunks = {}

        for doc in self.corpus:

            doc_id = doc["doc_id"]
            content = doc["content"]

            chunks[f"{doc_id}_semantic"] = {
                "content": content,
                "source_doc": doc_id,
                "strategy": "semantic",
            }

        print(
            f"✅ Semantic chunking: {len(chunks)} chunks created"
        )

        return chunks

    def chunk_corpus_parent_child(
        self,
        sentences_per_child: int = 2,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parent-child chunking.

        Each document keeps its full parent document and
        additionally creates smaller child chunks.
        """

        chunks = {}
        parent_chunk_id = 0

        for doc in self.corpus:

            doc_id = doc["doc_id"]
            content = doc["content"]

            # -------------------------
            # Parent
            # -------------------------

            parent_id = (
                f"{doc_id}_parent_{parent_chunk_id}"
            )

            chunks[parent_id] = {
                "content": content,
                "source_doc": doc_id,
                "strategy": "parent-child",
                "type": "parent",
            }

            # -------------------------
            # Children
            # -------------------------

            sentences = content.split(". ")

            for i in range(
                0,
                len(sentences),
                sentences_per_child,
            ):

                child_sentences = sentences[
                    i : i + sentences_per_child
                ]

                child_content = ". ".join(
                    child_sentences
                )

                if child_content.strip():

                    child_id = (
                        f"{doc_id}_child_"
                        f"{i // sentences_per_child}"
                    )

                    chunks[child_id] = {
                        "content": child_content,
                        "source_doc": doc_id,
                        "strategy": "parent-child",
                        "type": "child",
                        "parent_id": parent_id,
                    }

            parent_chunk_id += 1

        print(
            f"✅ Parent-child chunking: "
            f"{len(chunks)} chunks created"
        )

        return chunks

    # ============================================================
    # RETRIEVER
    # ============================================================

    def build_retriever(
        self,
        chunks: Dict[str, Dict[str, Any]],
        collection_name: str,
    ) -> Any:
        """Build ChromaDB collection."""

        try:
            self.chroma_client.delete_collection(
                name=collection_name
            )
        except Exception:
            pass

        collection = self.chroma_client.create_collection(
            name=collection_name
        )

        chunk_ids = []
        chunk_contents = []
        chunk_metadatas = []

        for chunk_id, chunk_data in chunks.items():

            chunk_ids.append(chunk_id)
            chunk_contents.append(
                chunk_data["content"]
            )

            chunk_metadatas.append(
                {
                    "source_doc": chunk_data["source_doc"],
                    "strategy": chunk_data["strategy"],
                }
            )

        embeddings = (
            self.embeddings_model
            .encode(chunk_contents)
            .tolist()
        )

        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_contents,
            metadatas=chunk_metadatas,
        )

        print(
            f"✅ ChromaDB collection "
            f"'{collection_name}' created with "
            f"{len(chunk_ids)} chunks"
        )

        return collection

    def retrieve(
        self,
        query: str,
        collection: Any,
        top_k: int = 3,
    ) -> List[str]:
        """Retrieve top-k documents."""

        query_embedding = (
            self.embeddings_model
            .encode(query)
            .tolist()
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        if not results["documents"]:
            return []

        return results["documents"][0]

    # ============================================================
    # ANSWER GENERATION
    # ============================================================

    def generate_answer(
        self,
        query: str,
        context: List[str],
    ) -> str:
        """Generate an answer using Groq."""

        context_text = "\n\n".join(context)

        prompt = f"""Answer based ONLY on this context.
Be concise.

Question:
{query}

Context:
{context_text}

Answer:"""

        response = (
            self.groq_client
            .chat.completions.create(
                model="openai/gpt-oss-120b",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        )

        text = response.choices[0].message.content

        if text is None:
            raise ValueError(
                "LLM returned empty answer."
            )

        return text.strip()

    # ============================================================
    # SCORE EXTRACTION
    # ============================================================

    @staticmethod
    def extract_score(text: str) -> float:
        """
        Extract a valid score between 0 and 1.
        """

        if not text:
            raise ValueError(
                "Empty LLM scoring response."
            )

        text = text.strip()

        # First try the entire response.
        try:

            score = float(text)

            if 0.0 <= score <= 1.0:
                return score

        except ValueError:
            pass

        # If the model returned something like:
        #
        # "Score: 0.92"
        #
        # extract the numeric score.

        match = re.search(
            r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)",
            text,
        )

        if match:

            score = float(match.group())

            return score

        raise ValueError(
            "Could not extract valid score from "
            f"LLM response: {text!r}"
        )

    # ============================================================
    # REUSABLE LLM SCORER
    # ============================================================

    def _call_llm_score(
        self,
        prompt: str,
        metric_name: str,
    ) -> Optional[float]:
        """
        Call Groq to score a metric.

        Returns None if the LLM response is invalid
        instead of pretending the score is 0.5.
        """

        try:

            response = (
                self.groq_client
                .chat.completions.create(
                    model="openai/gpt-oss-120b",
                    max_tokens=100,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )
            )

            text = response.choices[0].message.content

            if text is None:

                raise ValueError(
                    "LLM returned None content."
                )

            text = text.strip()

            print(
                f"DEBUG {metric_name.upper()} "
                f"RESPONSE: {repr(text)}"
            )

            return self.extract_score(text)

        except Exception as e:

            print(
                f"❌ {metric_name.upper()} ERROR: {e}"
            )

            return None

    # ============================================================
    # FAITHFULNESS
    # ============================================================

    def score_faithfulness(
        self,
        query: str,
        context: List[str],
        answer: str,
    ) -> Optional[float]:
        """Score whether answer is grounded in context."""

        context_text = "\n\n".join(context)

        prompt = f"""Return ONLY a number between 0 and 1.

Do not explain your answer.

Rate faithfulness:
Is the answer grounded in the provided context?

Context:
{context_text}

Answer:
{answer}

Score (0-1):"""

        return self._call_llm_score(
            prompt,
            "faithfulness",
        )

    # ============================================================
    # CONTEXT RELEVANCY
    # ============================================================

    def score_context_relevancy(
        self,
        query: str,
        context: List[str],
    ) -> Optional[float]:
        """Score relevance of retrieved context."""

        context_text = "\n\n".join(context)

        prompt = f"""Return ONLY a number between 0 and 1.

Do not explain your answer.

Rate context relevancy:
Is the retrieved context relevant to the query?

Query:
{query}

Context:
{context_text}

Score (0-1):"""

        return self._call_llm_score(
            prompt,
            "context_relevancy",
        )

    # ============================================================
    # ANSWER RELEVANCY
    # ============================================================

    def score_answer_relevancy(
        self,
        query: str,
        answer: str,
    ) -> Optional[float]:
        """Score whether answer addresses the query."""

        prompt = f"""Return ONLY a number between 0 and 1.

Do not explain your answer.

Rate answer relevancy:
Does the answer directly address the query?

Query:
{query}

Answer:
{answer}

Score (0-1):"""

        return self._call_llm_score(
            prompt,
            "answer_relevancy",
        )

    # ============================================================
    # CONTEXT RECALL
    # ============================================================

    def score_context_recall(
        self,
        query: str,
        context: List[str],
        answer: str,
    ) -> Optional[float]:
        """
        Score whether context contains the information
        needed to answer the question.
        """

        context_text = "\n\n".join(context)

        prompt = f"""Return ONLY a number between 0 and 1.

Do not explain your answer.

Rate context recall:
Does the context contain the information needed
to answer the question correctly?

Query:
{query}

Context:
{context_text}

Answer:
{answer}

Score (0-1):"""

        return self._call_llm_score(
            prompt,
            "context_recall",
        )

    # ============================================================
    # EVALUATION
    # ============================================================

    def run_evaluation(
        self,
        config: ChunkingConfig,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Run evaluation for one chunking strategy."""

        print("\n" + "=" * 60)
        print(f"🚀 Evaluating: {config.name}")
        print("=" * 60)

        # -------------------------
        # Create chunks
        # -------------------------

        if config.strategy == "semantic":

            chunks = self.chunk_corpus_semantic()

        else:

            chunks = self.chunk_corpus_parent_child()

        # -------------------------
        # Build retriever
        # -------------------------

        collection = self.build_retriever(
            chunks,
            collection_name=config.name,
        )

        results = []

        # -------------------------
        # Evaluate queries
        # -------------------------

        for i, test_query in enumerate(
            self.queries,
            1,
        ):

            query = test_query["query"]

            relevant_ids = test_query[
                "relevant_ids"
            ]

            print(
                f"\n[{i}/{len(self.queries)}] "
                f"{query[:60]}..."
            )

            # Retrieval
            retrieved_context = self.retrieve(
                query,
                collection,
                top_k=top_k,
            )

            # Answer generation
            answer = self.generate_answer(
                query,
                retrieved_context,
            )

            # Metrics
            faithfulness = (
                self.score_faithfulness(
                    query,
                    retrieved_context,
                    answer,
                )
            )

            context_relevancy = (
                self.score_context_relevancy(
                    query,
                    retrieved_context,
                )
            )

            answer_relevancy = (
                self.score_answer_relevancy(
                    query,
                    answer,
                )
            )

            context_recall = (
                self.score_context_recall(
                    query,
                    retrieved_context,
                    answer,
                )
            )

            # -------------------------
            # Save result
            # -------------------------

            result = {
                "query_id": test_query["query_id"],
                "query": query,
                "relevant_ids": relevant_ids,
                "retrieved_context": retrieved_context,
                "generated_answer": answer,
                "strategy": config.strategy,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_relevancy": context_relevancy,
                "context_recall": context_recall,
            }

            results.append(result)

            # -------------------------
            # Display scores
            # -------------------------

            f_score = (
                f"{faithfulness:.2f}"
                if faithfulness is not None
                else "N/A"
            )

            ar_score = (
                f"{answer_relevancy:.2f}"
                if answer_relevancy is not None
                else "N/A"
            )

            cr_score = (
                f"{context_relevancy:.2f}"
                if context_relevancy is not None
                else "N/A"
            )

            rec_score = (
                f"{context_recall:.2f}"
                if context_recall is not None
                else "N/A"
            )

            print(
                f"   F:{f_score} | "
                f"AR:{ar_score} | "
                f"CR:{cr_score} | "
                f"REC:{rec_score}"
            )

        return results

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    def save_results(
        self,
        results: List[Dict[str, Any]],
        config: ChunkingConfig,
    ):
        """Save evaluation results to JSON."""

        output_file = (
            "results/results_"
            f"{config.strategy.replace('-', '_')}.json"
        )

        os.makedirs(
            "results",
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
        ) as f:

            json.dump(
                results,
                f,
                indent=2,
            )

        print(
            f"\n💾 Results saved to {output_file}"
        )


# ================================================================
# MAIN
# ================================================================

def main():
    """Compare semantic and parent-child chunking."""

    evaluator = RAGEvaluator()

    configs = [

        ChunkingConfig(
            name="semantic_chunking",
            strategy="semantic",
            chunk_size=512,
            overlap=0,
        ),

        ChunkingConfig(
            name="parent_child_chunking",
            strategy="parent-child",
            chunk_size=256,
            overlap=50,
        ),

    ]

    all_results = {}

    # ------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------

    for config in configs:

        results = evaluator.run_evaluation(
            config
        )

        all_results[
            config.strategy
        ] = results

        evaluator.save_results(
            results,
            config,
        )

    # ------------------------------------------------------------
    # Results comparison
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("📊 RESULTS COMPARISON")
    print("=" * 60)

    for strategy, results in all_results.items():

        print(f"\n{strategy.upper()}:")

        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_relevancy",
            "context_recall",
        ]

        for metric in metrics:

            scores = [
                result[metric]
                for result in results
                if result[metric] is not None
            ]

            if not scores:

                print(
                    f"  {metric}: "
                    f"N/A (0/{len(results)} valid)"
                )

                continue

            avg_score = (
                sum(scores) / len(scores)
            )

            min_score = min(scores)
            max_score = max(scores)

            print(
                f"  {metric}: "
                f"{avg_score:.3f} "
                f"(range: "
                f"{min_score:.3f}-"
                f"{max_score:.3f}, "
                f"valid: "
                f"{len(scores)}/"
                f"{len(results)})"
            )

    print("\n✅ Evaluation complete!")
    print("📄 Next: python analyze_results.py")


if __name__ == "__main__":
    main()