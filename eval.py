#!/usr/bin/env python3

"""
RAGBench Evaluation Runner

Compares semantic chunking vs parent-child chunking using:

- Retrieval metrics from the benchmark's relevant_ids:
  Hit@1, Hit@3, MRR

- LLM-based generation metrics using Groq:
  Faithfulness and Answer Relevancy

The benchmark does not contain reference answers, so
Context Precision and Context Recall are not used.
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer


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
        """Initialize evaluator with corpus and test queries."""

        self.corpus = self._load_json(corpus_path)
        self.queries = self._load_json(queries_path)

        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Export it before running eval.py."
            )

        self.groq_client = Groq(
            api_key=groq_api_key
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

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def chunk_corpus_semantic(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Semantic chunking.

        For this benchmark implementation each source document
        remains one chunk. The source document ID is retained
        in metadata.
        """

        chunks: Dict[str, Dict[str, Any]] = {}

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
        Parent-child chunking.

        Parent = full source document.
        Children = chunks containing two sentences.
        """

        chunks: Dict[str, Dict[str, Any]] = {}

        parent_chunk_id = 0

        for doc in self.corpus:

            doc_id = doc["doc_id"]
            content = doc["content"]

            parent_id = (
                f"{doc_id}_parent_{parent_chunk_id}"
            )

            chunks[parent_id] = {
                "content": content,
                "source_doc": doc_id,
                "strategy": "parent-child",
                "type": "parent",
            }

            sentences = content.split(". ")

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
        """Build a ChromaDB collection for retrieval."""

        try:
            self.chroma_client.delete_collection(
                name=collection_name
            )
        except Exception:
            pass

        collection = self.chroma_client.create_collection(
            name=collection_name
        )

        chunk_ids: List[str] = []
        chunk_contents: List[str] = []
        chunk_metadatas: List[Dict[str, str]] = []

        for chunk_id, chunk_data in chunks.items():

            chunk_ids.append(chunk_id)
            chunk_contents.append(chunk_data["content"])

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
    ) -> Dict[str, List[str]]:
        """
        Retrieve top-k chunks.

        Returns chunk IDs and source document IDs.

        This matters because test_queries.json labels relevance
        at source-document level, while parent-child retrieval
        returns child chunk IDs.
        """

        query_embedding = (
            self.embeddings_model
            .encode(query)
            .tolist()
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        documents = (
            results["documents"][0]
            if results.get("documents")
            else []
        )

        ids = (
            results["ids"][0]
            if results.get("ids")
            else []
        )

        metadatas = (
            results["metadatas"][0]
            if results.get("metadatas")
            else []
        )

        source_doc_ids = [
            str(metadata.get("source_doc"))
            for metadata in metadatas
            if metadata.get("source_doc") is not None
        ]

        return {
            "documents": documents,
            "ids": ids,
            "source_doc_ids": source_doc_ids,
        }

    def generate_answer(
        self,
        query: str,
        context: List[str],
    ) -> str:
        """Generate an answer using Groq from the retrieved context."""

        context_text = "\n\n".join(context)

        prompt = f"""
Answer the following question based only on the provided context.

Be concise, accurate, and faithful to the context.

If the context does not contain enough information, say so.

Question:
{query}

Context:
{context_text}

Answer:
"""

        response = (
            self.groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_completion_tokens=500,
            )
        )

        return response.choices[0].message.content.strip()

    def evaluate_rag_pair(
        self,
        query: str,
        context: List[str],
        answer: str,
    ) -> Dict[str, Optional[float]]:
        """
        Evaluate the generated answer using Groq as an LLM judge.

        Faithfulness:
            Measures whether the answer is supported by the retrieved context.

        Answer relevancy:
            Measures whether the answer actually addresses the question.

        Scores are normalized to [0, 1].
        """

        context_text = "\n\n".join(context)

        try:

            # ---------------------------------------------------------
            # FAITHFULNESS
            # ---------------------------------------------------------

            faithfulness_prompt = f"""
You are evaluating a RAG system.

Determine whether the generated answer is fully supported
by the retrieved context.

Question:
{query}

Retrieved Context:
{context_text}

Generated Answer:
{answer}

Give a score from 0 to 1.

1.0 = Every factual claim in the answer is supported by the context.
0.75 = Mostly supported, with a small unsupported detail.
0.50 = Partially supported.
0.25 = Mostly unsupported.
0.0 = Completely unsupported or contradicted.

Return ONLY the numeric score.
"""

            faith_response = (
                self.groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a strict RAG evaluation judge. "
                                "Return only a number between 0 and 1."
                            ),
                        },
                        {
                            "role": "user",
                            "content": faithfulness_prompt,
                        },
                    ],
                    temperature=0,
        max_completion_tokens=256,
        reasoning_effort="low",
        include_reasoning=False,
                )
            )

            faithfulness_score = self._parse_score(
                faith_response.choices[0].message.content
            )

            # ---------------------------------------------------------
            # ANSWER RELEVANCY
            # ---------------------------------------------------------

            relevancy_prompt = f"""
You are evaluating a RAG system.

Determine how directly the generated answer answers
the user's question.

Question:
{query}

Generated Answer:
{answer}

Give a score from 0 to 1.

1.0 = Directly and completely answers the question.
0.75 = Directly answers it but misses a minor point.
0.50 = Partially answers the question.
0.25 = Barely addresses the question.
0.0 = Does not answer the question.

Do NOT judge factual correctness here.
Judge only how relevant the answer is to the question.

Return ONLY the numeric score.
"""

            relevancy_response = (
                self.groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a strict answer-relevancy judge. "
                                "Return only a number between 0 and 1."
                            ),
                        },
                        {
                            "role": "user",
                            "content": relevancy_prompt,
                        },
                    ],
                     temperature=0,
        max_completion_tokens=256,
        reasoning_effort="low",
        include_reasoning=False,
                )
            )

            answer_relevancy_score = self._parse_score(
                relevancy_response.choices[0].message.content
            )

            return {
                "faithfulness": faithfulness_score,
                "answer_relevancy": answer_relevancy_score,
            }

        except Exception as e:

            print(
                f"⚠️ LLM evaluation failed for "
                f"query '{query[:50]}...': {e}"
            )

            return {
                "faithfulness": None,
                "answer_relevancy": None,
            }

    @staticmethod
    def _parse_score(value: str) -> float:
        """
        Extract a numeric score from an LLM response
        and clamp it to [0, 1].
        """

        import re

        match = re.search(
            r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b",
            value.strip(),
        )

        if not match:
            raise ValueError(
                f"Could not parse evaluation score: {value}"
            )

        score = float(match.group())

        return min(
            1.0,
            max(0.0, score),
        )

    @staticmethod
    def evaluate_retrieval(
        relevant_ids: List[str],
        retrieved_source_doc_ids: List[str],
        top_k: int = 3,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval against test_queries.json relevant_ids.

        - Hit@1: relevant source document is ranked first.
        - Hit@3: relevant source document appears in top 3.
        - MRR: reciprocal rank of the first relevant source document.
        """

        relevant_set = set(relevant_ids)

        hit_at_1 = (
            1.0
            if (
                retrieved_source_doc_ids
                and retrieved_source_doc_ids[0] in relevant_set
            )
            else 0.0
        )

        hit_at_k = (
            1.0
            if any(
                doc_id in relevant_set
                for doc_id in retrieved_source_doc_ids[:top_k]
            )
            else 0.0
        )

        reciprocal_rank = 0.0

        for rank, doc_id in enumerate(
            retrieved_source_doc_ids,
            start=1,
        ):
            if doc_id in relevant_set:
                reciprocal_rank = 1.0 / rank
                break

        return {
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_k,
            "mrr": reciprocal_rank,
        }

    def run_evaluation(
        self,
        config: ChunkingConfig,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Run full evaluation for one chunking strategy."""

        print("\n" + "=" * 60)
        print(f"🚀 Evaluating: {config.name}")
        print("=" * 60)

        if config.strategy == "semantic":
            chunks = self.chunk_corpus_semantic()
        else:
            chunks = self.chunk_corpus_parent_child()

        collection = self.build_retriever(
            chunks,
            collection_name=config.name,
        )

        results: List[Dict[str, Any]] = []

        for i, test_query in enumerate(self.queries):

            query = test_query["query"]
            relevant_ids = test_query["relevant_ids"]

            print(
                f"\n[{i + 1}/{len(self.queries)}] "
                f"{query[:60]}..."
            )

            retrieved = self.retrieve(
                query,
                collection,
                top_k=top_k,
            )

            retrieved_context = retrieved["documents"]
            retrieved_ids = retrieved["ids"]
            retrieved_source_doc_ids = (
                retrieved["source_doc_ids"]
            )

            retrieval_metrics = self.evaluate_retrieval(
                relevant_ids,
                retrieved_source_doc_ids,
                top_k=top_k,
            )

            answer = self.generate_answer(
                query,
                retrieved_context,
            )

            rag_metrics = self.evaluate_rag_pair(
                query,
                retrieved_context,
                answer,
            )

            result = {
                "query_id": test_query["query_id"],
                "query": query,
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "retrieved_source_doc_ids": (
                    retrieved_source_doc_ids
                ),
                "retrieved_context": retrieved_context,
                "generated_answer": answer,
                "strategy": config.strategy,
                **retrieval_metrics,
                **rag_metrics,
            }

            results.append(result)

            print(
                f"   Retrieved source docs: "
                f"{retrieved_source_doc_ids}"
            )

            print(
                f"   Hit@1: "
                f"{retrieval_metrics['hit_at_1']:.3f}"
            )

            print(
                f"   Hit@3: "
                f"{retrieval_metrics['hit_at_3']:.3f}"
            )

            print(
                f"   MRR: "
                f"{retrieval_metrics['mrr']:.3f}"
            )

            faith_score = rag_metrics["faithfulness"]
            relevancy_score = rag_metrics["answer_relevancy"]

            if faith_score is None:
                print("   Faithfulness: FAILED")
            else:
                print(
                    f"   Faithfulness: "
                    f"{faith_score:.3f}"
                )

            if relevancy_score is None:
                print("   Answer Relevancy: FAILED")
            else:
                print(
                    f"   Answer Relevancy: "
                    f"{relevancy_score:.3f}"
                )

        return results

    def save_results(
        self,
        results: List[Dict[str, Any]],
        config: ChunkingConfig,
    ) -> str:
        """Save evaluation results to JSON."""

        output_file = (
            f"results/"
            f"results_{config.strategy}.json"
        )

        os.makedirs(
            "results",
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
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


def print_metric_summary(
    strategy: str,
    results: List[Dict[str, Any]],
) -> None:
    """Print averages without treating failed metrics as 0."""

    print(f"\n{strategy.upper()}:")

    metrics_dict = defaultdict(list)

    for result in results:

        for metric in (
            "hit_at_1",
            "hit_at_3",
            "mrr",
            "faithfulness",
            "answer_relevancy",
        ):

            score = result.get(metric)

            if score is not None:
                metrics_dict[metric].append(
                    float(score)
                )

    for metric in (
        "hit_at_1",
        "hit_at_3",
        "mrr",
        "faithfulness",
        "answer_relevancy",
    ):

        scores = metrics_dict.get(metric, [])

        if not scores:

            print(
                f"  {metric}: "
                "No valid scores"
            )

            continue

        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        print(f"  {metric}:")
        print(
            f"    Average: "
            f"{avg_score:.3f}"
        )

        print(
            f"    Min: {min_score:.3f} | "
            f"Max: {max_score:.3f}"
        )


def main():
    """Run full evaluation comparing both chunking strategies."""

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

    all_results: Dict[
        str,
        List[Dict[str, Any]]
    ] = {}

    for config in configs:

        results = evaluator.run_evaluation(
            config,
            top_k=3,
        )

        all_results[config.strategy] = results

        evaluator.save_results(
            results,
            config,
        )

    print("\n" + "=" * 60)
    print("📊 RESULTS COMPARISON")
    print("=" * 60)

    for strategy, results in all_results.items():

        print_metric_summary(
            strategy,
            results,
        )

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()