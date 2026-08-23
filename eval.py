#!/usr/bin/env python3

"""
RAGBench Evaluation Runner

Compares:
    1. Semantic chunking
    2. Parent-child chunking

Retrieval metrics:
    - Hit@1
    - Hit@3
    - MRR

Generation metrics:
    - Faithfulness
    - Answer Relevancy

Default:
    python eval.py

Full benchmark:
    RUN_FULL_EVAL=1 python eval.py
"""

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

PARENT_CHILD_FAILED_IDS = [
    "q_08",
    "q_09",
    "q_13",
    "q_14",
    "q_17",
]

MODEL_NAME = "openai/gpt-oss-120b"

RESULTS_DIR = "results"

PARENT_CHILD_RESULTS = (
    f"{RESULTS_DIR}/results_parent-child.json"
)


# ============================================================
# CHUNKING CONFIG
# ============================================================

@dataclass
class ChunkingConfig:
    name: str
    strategy: str
    chunk_size: int
    overlap: int


# ============================================================
# RAG EVALUATOR
# ============================================================

class RAGEvaluator:

    def __init__(
        self,
        corpus_path: str = "corpus.json",
        queries_path: str = "test_queries.json",
    ):
        self.corpus = self._load_json(corpus_path)
        self.queries = self._load_json(queries_path)

        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set.\n"
                "Run:\n"
                "export GROQ_API_KEY='your_key'"
            )

        self.groq_client = Groq(
            api_key=groq_api_key
        )

        print("Loading embedding model...")

        self.embeddings_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.chroma_client = chromadb.Client()

        print(
            f"✅ Loaded {len(self.corpus)} documents"
        )

        print(
            f"✅ Loaded {len(self.queries)} test queries"
        )

    # ========================================================
    # JSON
    # ========================================================

    @staticmethod
    def _load_json(path: str) -> Any:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    # ========================================================
    # SEMANTIC CHUNKING
    # ========================================================

    def chunk_corpus_semantic(
        self,
    ) -> Dict[str, Dict[str, Any]]:

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

    # ========================================================
    # PARENT-CHILD CHUNKING
    # ========================================================

    def chunk_corpus_parent_child(
        self,
        sentences_per_child: int = 2,
    ) -> Dict[str, Dict[str, Any]]:

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

            # Correct sentence splitting
            sentences = re.split(
                r"(?<=[.!?])\s+",
                content.strip(),
            )

            for i in range(
                0,
                len(sentences),
                sentences_per_child,
            ):

                child_sentences = sentences[
                    i:i + sentences_per_child
                ]

                child_content = " ".join(
                    child_sentences
                ).strip()

                if not child_content:
                    continue

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

    # ========================================================
    # CHROMADB
    # ========================================================

    def build_retriever(
        self,
        chunks: Dict[str, Dict[str, Any]],
        collection_name: str,
    ) -> Any:

        try:
            self.chroma_client.delete_collection(
                name=collection_name
            )
        except Exception:
            pass

        collection = (
            self.chroma_client.create_collection(
                name=collection_name
            )
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

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query: str,
        collection: Any,
        top_k: int = 3,
    ) -> Dict[str, List[str]]:

        query_embedding = (
            self.embeddings_model
            .encode(query)
            .tolist()
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
            ],
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

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def generate_answer(
        self,
        query: str,
        context: List[str],
    ) -> str:

        context_text = "\n\n".join(context)

        prompt = f"""
Answer the following question using ONLY the
provided context.

Be concise, accurate, and faithful to the context.

If the context does not contain enough information,
say that the information is insufficient.

Question:
{query}

Context:
{context_text}

Answer:
"""

        response = (
            self.groq_client
            .chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_completion_tokens=300,
            )
        )

        content = response.choices[0].message.content

        return content.strip() if content else ""

    # ========================================================
    # SCORE PARSER
    # ========================================================

    @staticmethod
    def parse_score(raw: str) -> Optional[float]:

        if not raw:
            return None

        text = raw.strip()

        # Exact allowed values
        if text in {
            "0",
            "0.25",
            "0.5",
            "0.75",
            "1",
            "1.0",
        }:
            return float(text)

        # Look for a standalone allowed number
        match = re.search(
            r"(?<!\d)(1(?:\.0+)?|0(?:\.(?:25|5|75))?)(?!\d)",
            text,
        )

        if match:
            return float(match.group(1))

        return None

    # ========================================================
    # SINGLE METRIC JUDGE
    # ========================================================

    def judge_metric(
        self,
        metric_name: str,
        criteria: str,
        query: str,
        context: List[str],
        answer: str,
    ) -> Optional[float]:

        context_text = "\n\n".join(context)

        prompt = f"""
You are evaluating a RAG system.

QUESTION:
{query}

RETRIEVED CONTEXT:
{context_text}

GENERATED ANSWER:
{answer}

METRIC:
{metric_name}

CRITERIA:
{criteria}

Choose exactly one score:

0
0.25
0.5
0.75
1

Return ONLY the score.
Do not explain.
Do not use JSON.
Do not use Markdown.
Do not write any other text.

Your response:
"""

        try:
            response = (
                self.groq_client
                .chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return exactly one number: "
                                "0, 0.25, 0.5, 0.75, or 1."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    max_completion_tokens=20,
                    temperature=0,
                )
            )

            message = response.choices[0].message

            raw = (
                message.content
                if message.content
                else ""
            )

            raw = raw.strip()

            print(
                f"      {metric_name}: {raw!r}"
            )

            score = self.parse_score(raw)

            if score is not None:
                return max(
                    0.0,
                    min(1.0, score),
                )

        except Exception as e:

            print(
                f"      ⚠️ {metric_name} judge error: {e}"
            )

        return None

    # ========================================================
    # RAG JUDGE
    # ========================================================

    def evaluate_rag_pair(
        self,
        query: str,
        context: List[str],
        answer: str,
    ) -> Dict[str, Optional[float]]:

        faithfulness = self.judge_metric(
            metric_name="faithfulness",
            criteria="""
Score 1 when every factual claim in the answer
is directly supported by the retrieved context.

Score 0.75 when almost all claims are supported
and there is only a minor unsupported detail.

Score 0.5 when some important claims are supported
but other claims are unsupported.

Score 0.25 when most claims are unsupported.

Score 0 when the answer is unsupported or
contradicts the retrieved context.
""",
            query=query,
            context=context,
            answer=answer,
        )

        answer_relevancy = self.judge_metric(
            metric_name="answer_relevancy",
            criteria="""
Score 1 when the answer directly and completely
answers the user's question.

Score 0.75 when it answers the question with
only a minor omission.

Score 0.5 when it partially answers the question.

Score 0.25 when it barely addresses the question.

Score 0 when it does not answer the question.
""",
            query=query,
            context=context,
            answer=answer,
        )

        return {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        }

    # ========================================================
    # RETRIEVAL METRICS
    # ========================================================

    @staticmethod
    def evaluate_retrieval(
        relevant_ids: List[str],
        retrieved_source_doc_ids: List[str],
        top_k: int = 3,
    ) -> Dict[str, float]:

        relevant_set = set(relevant_ids)

        hit_at_1 = (
            1.0
            if (
                retrieved_source_doc_ids
                and retrieved_source_doc_ids[0]
                in relevant_set
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

    # ========================================================
    # FULL EVALUATION
    # ========================================================

    def run_evaluation(
        self,
        config: ChunkingConfig,
        top_k: int = 3,
        resume: bool = True,
    ) -> List[Dict[str, Any]]:

        print("\n" + "=" * 60)

        print(
            f"🚀 Evaluating: {config.name}"
        )

        print("=" * 60)

        if config.strategy == "semantic":
            chunks = self.chunk_corpus_semantic()
        else:
            chunks = self.chunk_corpus_parent_child()

        collection = self.build_retriever(
            chunks,
            config.name,
        )

        output_file = (
            f"{RESULTS_DIR}/"
            f"results_{config.strategy}.json"
        )

        results = []
        completed_ids = set()

        if resume and os.path.exists(output_file):

            try:

                with open(
                    output_file,
                    "r",
                    encoding="utf-8",
                ) as f:
                    previous = json.load(f)

                if isinstance(previous, list):

                    results = previous

                    completed_ids = {
                        r.get("query_id")
                        for r in results
                        if (
                            r.get("faithfulness")
                            is not None
                            and
                            r.get("answer_relevancy")
                            is not None
                        )
                    }

                    print(
                        f"↩️ Resuming: "
                        f"{len(results)} saved queries"
                    )

            except Exception as e:

                print(
                    f"⚠️ Could not load "
                    f"previous results: {e}"
                )

        for i, test_query in enumerate(
            self.queries
        ):

            query_id = test_query["query_id"]

            if query_id in completed_ids:

                print(
                    f"\n[{i + 1}/"
                    f"{len(self.queries)}] "
                    f"{query_id} SKIPPED"
                )

                continue

            query = test_query["query"]

            relevant_ids = test_query[
                "relevant_ids"
            ]

            print(
                f"\n[{i + 1}/"
                f"{len(self.queries)}] "
                f"{query[:60]}..."
            )

            retrieved = self.retrieve(
                query,
                collection,
                top_k=top_k,
            )

            retrieved_context = retrieved[
                "documents"
            ]

            retrieved_ids = retrieved["ids"]

            retrieved_source_doc_ids = (
                retrieved["source_doc_ids"]
            )

            retrieval_metrics = (
                self.evaluate_retrieval(
                    relevant_ids,
                    retrieved_source_doc_ids,
                    top_k=top_k,
                )
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
                "query_id": query_id,
                "query": query,
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "retrieved_source_doc_ids":
                    retrieved_source_doc_ids,
                "retrieved_context":
                    retrieved_context,
                "generated_answer": answer,
                "strategy": config.strategy,
                **retrieval_metrics,
                **rag_metrics,
            }

            existing_index = next(
                (
                    index
                    for index, item in enumerate(results)
                    if item.get("query_id") == query_id
                ),
                None,
            )

            if existing_index is not None:
                results[existing_index] = result
            else:
                results.append(result)

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

            faith = rag_metrics["faithfulness"]

            relevancy = rag_metrics[
                "answer_relevancy"
            ]

            print(
                f"   Faithfulness: "
                f"{faith:.3f}"
                if faith is not None
                else
                "   Faithfulness: FAILED"
            )

            print(
                f"   Answer Relevancy: "
                f"{relevancy:.3f}"
                if relevancy is not None
                else
                "   Answer Relevancy: FAILED"
            )

            os.makedirs(
                RESULTS_DIR,
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

        return results

    # ========================================================
    # FINAL PARENT-CHILD REPAIR
    # ========================================================

    def retry_failed_parent_child(
        self,
        failed_query_ids: List[str],
    ) -> List[Dict[str, Any]]:

        if not os.path.exists(
            PARENT_CHILD_RESULTS
        ):

            raise FileNotFoundError(
                f"Could not find "
                f"{PARENT_CHILD_RESULTS}"
            )

        with open(
            PARENT_CHILD_RESULTS,
            "r",
            encoding="utf-8",
        ) as f:

            results = json.load(f)

        target_ids = set(
            failed_query_ids
        )

        print("\n" + "=" * 60)

        print(
            "🔧 FINAL PARENT-CHILD JUDGE REPAIR"
        )

        print("=" * 60)

        print(
            f"Target IDs: "
            f"{sorted(target_ids)}"
        )

        attempted = 0
        repaired = 0

        for result in results:

            query_id = result.get(
                "query_id"
            )

            if query_id not in target_ids:
                continue

            if (
                result.get("faithfulness")
                is not None
                and
                result.get("answer_relevancy")
                is not None
            ):

                print(
                    f"\n✅ {query_id} "
                    f"already repaired. Skipping."
                )

                continue

            attempted += 1

            print("\n" + "-" * 60)

            print(
                f"🔁 Evaluating {query_id}"
            )

            print(
                f"Question: "
                f"{result.get('query', '')}"
            )

            context = result.get(
                "retrieved_context",
                [],
            )

            answer = result.get(
                "generated_answer",
                "",
            )

            if not context:

                print(
                    "⚠️ No stored context."
                )

                continue

            if not answer:

                print(
                    "⚠️ No stored answer."
                )

                continue

            print(
                "   ♻️ Reusing stored "
                "context and answer"
            )

            rag_metrics = self.evaluate_rag_pair(
                result["query"],
                context,
                answer,
            )

            faith = rag_metrics[
                "faithfulness"
            ]

            relevancy = rag_metrics[
                "answer_relevancy"
            ]

            if faith is not None:
                result["faithfulness"] = faith

            if relevancy is not None:
                result["answer_relevancy"] = relevancy

            if (
                faith is not None
                and relevancy is not None
            ):

                repaired += 1

                print(
                    "   ✅ Both scores repaired"
                )

            else:

                print(
                    "   ⚠️ Judge failed to return "
                    "both scores"
                )

            print(
                f"   Faithfulness: "
                f"{result.get('faithfulness')}"
            )

            print(
                f"   Answer Relevancy: "
                f"{result.get('answer_relevancy')}"
            )

            os.makedirs(
                RESULTS_DIR,
                exist_ok=True,
            )

            with open(
                PARENT_CHILD_RESULTS,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    results,
                    f,
                    indent=2,
                )

            print("   💾 Saved")

        print("\n" + "=" * 60)

        print(
            f"📌 Retry attempts: {attempted}"
        )

        print(
            f"✅ Successfully repaired: {repaired}"
        )

        print("=" * 60)

        return results


# ============================================================
# METRIC SUMMARY
# ============================================================

def print_metric_summary(
    strategy: str,
    results: List[Dict[str, Any]],
) -> None:

    print(
        f"\n{strategy.upper()}:"
    )

    metrics_dict = defaultdict(list)

    metrics = [
        "hit_at_1",
        "hit_at_3",
        "mrr",
        "faithfulness",
        "answer_relevancy",
    ]

    for result in results:

        for metric in metrics:

            score = result.get(metric)

            if score is not None:

                metrics_dict[
                    metric
                ].append(
                    float(score)
                )

    for metric in metrics:

        scores = metrics_dict.get(
            metric,
            [],
        )

        if not scores:

            print(
                f"  {metric}: "
                "No valid scores"
            )

            continue

        average = (
            sum(scores)
            / len(scores)
        )

        print(
            f"  {metric}:"
        )

        print(
            f"    Average: "
            f"{average:.3f}"
        )

        print(
            f"    Min: "
            f"{min(scores):.3f} | "
            f"Max: "
            f"{max(scores):.3f}"
        )

        print(
            f"    Valid scores: "
            f"{len(scores)}/{len(results)}"
        )


# ============================================================
# RETRY MODE
# ============================================================

def retry_mode():

    evaluator = RAGEvaluator()

    results = evaluator.retry_failed_parent_child(
        PARENT_CHILD_FAILED_IDS
    )

    print("\n" + "=" * 60)

    print(
        "📊 UPDATED PARENT-CHILD RESULTS"
    )

    print("=" * 60)

    print_metric_summary(
        "parent-child",
        results,
    )

    remaining_failures = [
        r.get("query_id")
        for r in results
        if (
            r.get("query_id")
            in PARENT_CHILD_FAILED_IDS
            and (
                r.get("faithfulness") is None
                or
                r.get("answer_relevancy") is None
            )
        )
    ]

    print()

    if remaining_failures:

        print(
            "⚠️ Still failed:"
        )

        for query_id in remaining_failures:

            print(
                f"   - {query_id}"
            )

        print(
            "\nThese are judge failures, "
            "not retrieval failures."
        )

    else:

        print(
            "🎉 All targeted "
            "parent-child evaluations "
            "are complete!"
        )


# ============================================================
# FULL EVALUATION MODE
# ============================================================

def full_evaluation_mode():

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

    # --------------------------------------------------------
    # SEMANTIC
    # --------------------------------------------------------

    semantic_file = (
        f"{RESULTS_DIR}/"
        f"results_semantic.json"
    )

    rerun_semantic = (
        os.getenv(
            "RERUN_SEMANTIC",
            "0",
        )
        == "1"
    )

    if (
        os.path.exists(semantic_file)
        and not rerun_semantic
    ):

        with open(
            semantic_file,
            "r",
            encoding="utf-8",
        ) as f:

            semantic_results = json.load(f)

        all_results["semantic"] = (
            semantic_results
        )

        print(
            f"↩️ Reusing existing "
            f"semantic results: "
            f"{len(semantic_results)} queries"
        )

    else:

        semantic_results = evaluator.run_evaluation(
            configs[0],
            top_k=3,
            resume=True,
        )

        all_results["semantic"] = (
            semantic_results
        )

    # --------------------------------------------------------
    # PARENT-CHILD
    # --------------------------------------------------------

    parent_results = evaluator.run_evaluation(
        configs[1],
        top_k=3,
        resume=True,
    )

    all_results["parent-child"] = (
        parent_results
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "📊 RESULTS COMPARISON"
    )

    print("=" * 60)

    for strategy, results in (
        all_results.items()
    ):

        print_metric_summary(
            strategy,
            results,
        )

    print(
        "\n✅ Evaluation complete!"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    run_full_eval = (
        os.getenv(
            "RUN_FULL_EVAL",
            "0",
        )
        == "1"
    )

    if run_full_eval:

        full_evaluation_mode()

    else:

        retry_mode()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()