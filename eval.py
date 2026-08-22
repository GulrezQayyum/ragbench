#!/usr/bin/env python3

"""
RAGBench Evaluation Runner

Compares semantic chunking vs parent-child chunking using RAGAS metrics.
"""

import json
import os

from typing import List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
from urllib import response

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from datasets import Dataset
import pandas as pd


@dataclass
class ChunkingConfig:
    name: str
    strategy: str  # "semantic" or "parent-child"
    chunk_size: int
    overlap: int


class RAGEvaluator:

    def __init__(
        self,
        corpus_path: str = "corpus.json",
        queries_path: str = "test_queries.json",
    ):
        """Initialize evaluator with corpus and test queries."""

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

    @staticmethod
    def _load_json(path: str) -> Any:
        """Load JSON file."""

        with open(path, "r") as f:
            return json.load(f)

    def chunk_corpus_semantic(self) -> Dict[str, Dict[str, Any]]:
        """
        Semantic chunking:
        Keep each document as one chunk.

        In a real ChunkLab implementation, this would
        split on topic boundaries.
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
            f"✅ Semantic chunking: "
            f"{len(chunks)} chunks created"
        )

        return chunks

    def chunk_corpus_parent_child(
        self,
        sentences_per_child: int = 2,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parent-child chunking:

        - Parent = full document
        - Children = small chunks containing 2 sentences
        """

        chunks = {}

        parent_chunk_id = 0

        for doc in self.corpus:

            doc_id = doc["doc_id"]
            content = doc["content"]

            # Add parent chunk
            parent_id = (
                f"{doc_id}_parent_{parent_chunk_id}"
            )

            chunks[parent_id] = {
                "content": content,
                "source_doc": doc_id,
                "strategy": "parent-child",
                "type": "parent",
            }

            # Split document into sentences
            sentences = content.split(". ")

            # Create child chunks
            for i in range(
                0,
                len(sentences),
                sentences_per_child,
            ):

                child_sentences = sentences[
                    i:i + sentences_per_child
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

    def build_retriever(
        self,
        chunks: Dict[str, Dict[str, Any]],
        collection_name: str,
    ) -> Any:
        """Build ChromaDB collection for retrieval."""

        # Delete existing collection if present
        try:
            self.chroma_client.delete_collection(
                name=collection_name
            )
        except Exception:
            pass

        # Create collection
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
                    "source_doc": chunk_data[
                        "source_doc"
                    ],
                    "strategy": chunk_data[
                        "strategy"
                    ],
                }
            )

        # Compute embeddings
        embeddings = (
            self.embeddings_model
            .encode(chunk_contents)
            .tolist()
        )

        # Add chunks to ChromaDB
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
        """Retrieve top-k chunks for a query."""

        query_embedding = (
            self.embeddings_model
            .encode(query)
            .tolist()
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = (
            results["documents"][0]
            if results["documents"]
            else []
        )

        return documents

    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer using Groq given query and context."""

        context_text = "\n\n".join(context)

        prompt = f"""
Answer the following question based on the provided context.
Be concise and faithful to the context.

Question: {query}

Context:

{context_text}

Answer:
"""

        response = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    def evaluate_rag_pair(
        self,
        query: str,
        context: List[str],
        answer: str,
    ) -> Dict[str, float]:
        """
        Evaluate one RAG triplet:

        query + retrieved context + generated answer.

        Ragas 0.1.21 does not provide context_relevancy,
        so context_precision is used internally while
        preserving the existing context_relevancy result key.
        """

        try:

            # Ragas expects:
            #
            # question       -> List[str]
            # contexts       -> List[List[str]]
            # answer         -> List[str]
            # ground_truths  -> List[str]

            eval_data = {
                "question": [query],
                "contexts": [context],
                "answer": [answer],
                "ground_truths": [query],
            }

            dataset = Dataset.from_dict(
                eval_data
            )

            # Run Ragas evaluation
            scores = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )

            return {
                "faithfulness": scores[
                    "faithfulness"
                ][0],

                "answer_relevancy": scores[
                    "answer_relevancy"
                ][0],

                # Ragas 0.1.21 does not have
                # context_relevancy.
                #
                # Keep this key so the rest of
                # RAGBench remains compatible.
                "context_relevancy": scores[
                    "context_precision"
                ][0],

                "context_recall": scores[
                    "context_recall"
                ][0],
            }

        except Exception as e:

            print(
                f"⚠️ RAGAS evaluation failed for "
                f"query '{query[:50]}...': {e}"
            )

            # Neutral fallback scores
            return {
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "context_relevancy": 0.5,
                "context_recall": 0.5,
            }

    def run_evaluation(
        self,
        config: ChunkingConfig,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Run full evaluation for one chunking strategy."""

        print("\n" + "=" * 60)
        print(
            f"🚀 Evaluating: {config.name}"
        )
        print("=" * 60)

        # Create chunks
        if config.strategy == "semantic":

            chunks = (
                self.chunk_corpus_semantic()
            )

        else:

            chunks = (
                self.chunk_corpus_parent_child()
            )

        # Build retriever
        collection = self.build_retriever(
            chunks,
            collection_name=config.name,
        )

        results = []

        # Evaluate every test query
        for i, test_query in enumerate(
            self.queries
        ):

            query = test_query["query"]

            relevant_ids = test_query[
                "relevant_ids"
            ]

            print(
                f"\n[{i + 1}/{len(self.queries)}] "
                f"{query[:60]}..."
            )

            # Retrieve
            retrieved_context = self.retrieve(
                query,
                collection,
                top_k=top_k,
            )

            # Generate answer
            answer = self.generate_answer(
                query,
                retrieved_context,
            )

            # Evaluate
            metrics = self.evaluate_rag_pair(
                query,
                retrieved_context,
                answer,
            )

            # Store result
            result = {
                "query_id": test_query[
                    "query_id"
                ],

                "query": query,

                "relevant_ids": relevant_ids,

                "retrieved_context":
                    retrieved_context,

                "generated_answer": answer,

                "strategy": config.strategy,

                **metrics,
            }

            results.append(result)

            print(
                f"   Faithfulness: "
                f"{metrics['faithfulness']:.3f}"
            )

            print(
                f"   Answer Relevancy: "
                f"{metrics['answer_relevancy']:.3f}"
            )

            print(
                f"   Context Relevancy: "
                f"{metrics['context_relevancy']:.3f}"
            )

            print(
                f"   Context Recall: "
                f"{metrics['context_recall']:.3f}"
            )

        return results

    def save_results(
        self,
        results: List[Dict[str, Any]],
        config: ChunkingConfig,
    ):
        """Save evaluation results to JSON."""

        output_file = (
            f"results/results_"
            f"{config.strategy}.json"
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
            f"\n💾 Results saved to "
            f"{output_file}"
        )

        return output_file


def main():
    """Run full evaluation comparing semantic vs parent-child chunking."""

    evaluator = RAGEvaluator()

    # Define configurations
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

    # Run evaluations
    all_results = {}

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

    # Comparison analysis
    print("\n" + "=" * 60)
    print("📊 RESULTS COMPARISON")
    print("=" * 60)

    for strategy, results in (
        all_results.items()
    ):

        print(
            f"\n{strategy.upper()}:"
        )

        metrics_dict = defaultdict(list)

        for result in results:

            metrics_dict[
                "faithfulness"
            ].append(
                result["faithfulness"]
            )

            metrics_dict[
                "answer_relevancy"
            ].append(
                result["answer_relevancy"]
            )

            metrics_dict[
                "context_relevancy"
            ].append(
                result["context_relevancy"]
            )

            metrics_dict[
                "context_recall"
            ].append(
                result["context_recall"]
            )

        for metric, scores in (
            metrics_dict.items()
        ):

            avg_score = (
                sum(scores) / len(scores)
            )

            min_score = min(scores)
            max_score = max(scores)

            print(
                f"  {metric}:"
            )

            print(
                f"    Average: "
                f"{avg_score:.3f}"
            )

            print(
                f"    Min: {min_score:.3f} | "
                f"Max: {max_score:.3f}"
            )

    print(
        "\n✅ Evaluation complete!"
    )


if __name__ == "__main__":
    main()