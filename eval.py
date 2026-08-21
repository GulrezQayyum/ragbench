#!/usr/bin/env python3
"""
RAGBench Evaluation Runner - FIXED VERSION
- Better score extraction with more robust parsing
- Improved prompts for Mixtral
- Better error handling and debugging
"""

import json
import os
import re
import time
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

    @staticmethod
    def _load_json(path: str) -> Any:
        """Load JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def chunk_corpus_semantic(self) -> Dict[str, Dict[str, Any]]:
        """Semantic chunking: each document as one chunk."""
        chunks = {}
        for doc in self.corpus:
            doc_id = doc["doc_id"]
            content = doc["content"]
            chunks[f"{doc_id}_semantic"] = {
                "content": content,
                "source_doc": doc_id,
                "strategy": "semantic",
            }
        print(f"✅ Semantic chunking: {len(chunks)} chunks created")
        return chunks

    def chunk_corpus_parent_child(
        self,
        sentences_per_child: int = 2,
    ) -> Dict[str, Dict[str, Any]]:
        """Parent-child chunking: small children + full parent."""
        chunks = {}
        parent_chunk_id = 0

        for doc in self.corpus:
            doc_id = doc["doc_id"]
            content = doc["content"]

            parent_id = f"{doc_id}_parent_{parent_chunk_id}"
            chunks[parent_id] = {
                "content": content,
                "source_doc": doc_id,
                "strategy": "parent-child",
                "type": "parent",
            }

            sentences = content.split(". ")
            for i in range(0, len(sentences), sentences_per_child):
                child_sentences = sentences[i : i + sentences_per_child]
                child_content = ". ".join(child_sentences)

                if child_content.strip():
                    child_id = f"{doc_id}_child_{i // sentences_per_child}"
                    chunks[child_id] = {
                        "content": child_content,
                        "source_doc": doc_id,
                        "strategy": "parent-child",
                        "type": "child",
                        "parent_id": parent_id,
                    }

            parent_chunk_id += 1

        print(f"✅ Parent-child chunking: {len(chunks)} chunks created")
        return chunks

    def build_retriever(
        self,
        chunks: Dict[str, Dict[str, Any]],
        collection_name: str,
    ) -> Any:
        """Build ChromaDB collection."""
        try:
            self.chroma_client.delete_collection(name=collection_name)
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
            chunk_contents.append(chunk_data["content"])
            chunk_metadatas.append({
                "source_doc": chunk_data["source_doc"],
                "strategy": chunk_data["strategy"],
            })

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

        print(f"✅ ChromaDB collection '{collection_name}' created with {len(chunk_ids)} chunks")
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

    def generate_answer(
        self,
        query: str,
        context: List[str],
    ) -> str:
        """Generate answer using openai/gpt-oss-120b."""
        context_text = "\n\n".join(context)

        prompt = f"""Answer based ONLY on context. Be concise.

Question: {query}

Context: {context_text}

Answer:"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = (
                    self.groq_client
                    .chat.completions.create(
                        model="openai/gpt-oss-120b",
                        max_tokens=200,
                        messages=[{"role": "user", "content": prompt}],
                    )
                )

                text = response.choices[0].message.content
                if text is None:
                    raise ValueError("LLM returned empty answer.")
                return text.strip()

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"   ⚠️ Generation retry {attempt + 1}/{max_retries}: {str(e)[:50]}")
                    time.sleep(2)
                else:
                    raise

    @staticmethod
    def extract_score(text: str) -> float:
        """
        Extract score 0-1 from text with more robust parsing.
        Handles various response formats from Mixtral.
        """
        if not text or not text.strip():
            raise ValueError("Empty response")

        text = text.strip().lower()
        
        # Try to find any number between 0 and 1
        # Pattern matches: 0.85, .75, 0, 1, 0.0, 1.0
        patterns = [
            r'(\b0\.\d+\b)',  # 0.85
            r'(\b\.\d+\b)',   # .75
            r'(\b[01]\b)',    # 0 or 1
            r'(\b0\.0\b)',    # 0.0
            r'(\b1\.0\b)',    # 1.0
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    score = float(match.group(1))
                    if 0.0 <= score <= 1.0:
                        return score
                except ValueError:
                    continue
        
        # Try to find percentage (e.g., "85%")
        percent_match = re.search(r'(\d+)%', text)
        if percent_match:
            try:
                score = float(percent_match.group(1)) / 100.0
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                pass
        
        # Try to find score in text like "score: 0.85"
        score_match = re.search(r'score:?\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                pass
        
        # Check for explicit yes/no or true/false
        if re.search(r'\byes\b|true', text):
            return 1.0
        if re.search(r'\bno\b|false', text):
            return 0.0
        
        raise ValueError(f"Could not parse score from: {text[:100]}")

    def _call_llm_score(
        self,
        prompt: str,
        metric_name: str,
    ) -> Optional[float]:
        """Call mixtral to score a metric (with retries and delays)."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Add delay to avoid rate limiting
                if attempt > 0:
                    time.sleep(2 ** attempt)  # Exponential backoff

                response = (
                    self.groq_client
                    .chat.completions.create(
                        model="mixtral-8x7b-32768",
                        max_tokens=100,
                        temperature=0.1,  # Lower temperature for more consistent scoring
                        messages=[{"role": "user", "content": prompt}],
                    )
                )

                text = response.choices[0].message.content
                if text is None or text.strip() == "":
                    if attempt < max_retries - 1:
                        continue
                    return None

                # Debug: print raw response for first few queries
                # Uncomment for debugging:
                # print(f"   Debug - {metric_name} response: {text[:100]}")

                return self.extract_score(text)

            except Exception as e:
                if "rate_limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        print(f"   ⏳ Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"   ⚠️ {metric_name} failed after retries (rate limit)")
                        return None
                else:
                    if attempt < max_retries - 1:
                        print(f"   ⚠️ {metric_name} retry {attempt + 1}/{max_retries}: {str(e)[:50]}")
                        time.sleep(1)
                    else:
                        print(f"   ⚠️ {metric_name} failed: {str(e)[:50]}")
                        return None

        return None

    def score_faithfulness(
        self,
        query: str,
        context: List[str],
        answer: str,
    ) -> Optional[float]:
        """Score if answer is grounded in context (0-1)."""
        context_snippet = " ".join(context)[:300]
        answer_snippet = answer[:200]

        prompt = f"""Rate from 0.0 to 1.0 how much the answer is grounded in (supported by) the provided context.
Return only a single number between 0.0 and 1.0.

Context: {context_snippet}

Answer: {answer_snippet}

Score (0.0-1.0):"""

        return self._call_llm_score(prompt, "faithfulness")

    def score_context_relevancy(
        self,
        query: str,
        context: List[str],
    ) -> Optional[float]:
        """Score if context is relevant to query (0-1)."""
        context_snippet = " ".join(context)[:300]

        prompt = f"""Rate from 0.0 to 1.0 how relevant the context is to answering the query.
Return only a single number between 0.0 and 1.0.

Query: {query}

Context: {context_snippet}

Score (0.0-1.0):"""

        return self._call_llm_score(prompt, "context_relevancy")

    def score_answer_relevancy(
        self,
        query: str,
        answer: str,
    ) -> Optional[float]:
        """Score if answer addresses query (0-1)."""
        answer_snippet = answer[:200]

        prompt = f"""Rate from 0.0 to 1.0 how well the answer directly addresses the query.
Return only a single number between 0.0 and 1.0.

Query: {query}

Answer: {answer_snippet}

Score (0.0-1.0):"""

        return self._call_llm_score(prompt, "answer_relevancy")

    def score_context_recall(
        self,
        query: str,
        context: List[str],
        answer: str,
    ) -> Optional[float]:
        """Score if context has info needed for answer (0-1)."""
        context_snippet = " ".join(context)[:300]
        answer_snippet = answer[:200]

        prompt = f"""Rate from 0.0 to 1.0 whether the context contains enough information to support the answer.
Return only a single number between 0.0 and 1.0.

Query: {query}

Context: {context_snippet}

Answer: {answer_snippet}

Score (0.0-1.0):"""

        return self._call_llm_score(prompt, "context_recall")

    def run_evaluation(
        self,
        config: ChunkingConfig,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Run evaluation for one chunking strategy."""

        print("\n" + "=" * 60)
        print(f"🚀 Evaluating: {config.name}")
        print("=" * 60)

        if config.strategy == "semantic":
            chunks = self.chunk_corpus_semantic()
        else:
            chunks = self.chunk_corpus_parent_child()

        collection = self.build_retriever(chunks, collection_name=config.name)

        results = []

        for i, test_query in enumerate(self.queries, 1):
            query = test_query["query"]
            relevant_ids = test_query["relevant_ids"]

            print(f"\n[{i}/{len(self.queries)}] {query[:60]}...")

            try:
                retrieved_context = self.retrieve(query, collection, top_k=top_k)
                
                # Skip if no context retrieved
                if not retrieved_context:
                    print(f"   ⚠️ No context retrieved, skipping query")
                    continue
                
                answer = self.generate_answer(query, retrieved_context)
                
                # Add small delay between scoring calls to avoid rate limits
                faithfulness = self.score_faithfulness(query, retrieved_context, answer)
                time.sleep(0.5)
                
                context_relevancy = self.score_context_relevancy(query, retrieved_context)
                time.sleep(0.5)
                
                answer_relevancy = self.score_answer_relevancy(query, answer)
                time.sleep(0.5)
                
                context_recall = self.score_context_recall(query, retrieved_context, answer)
                time.sleep(0.5)

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

                f_score = f"{faithfulness:.2f}" if faithfulness is not None else "N/A"
                ar_score = f"{answer_relevancy:.2f}" if answer_relevancy is not None else "N/A"
                cr_score = f"{context_relevancy:.2f}" if context_relevancy is not None else "N/A"
                rec_score = f"{context_recall:.2f}" if context_recall is not None else "N/A"

                print(f"   F:{f_score} | AR:{ar_score} | CR:{cr_score} | REC:{rec_score}")

            except Exception as e:
                print(f"   ❌ Query error: {str(e)[:100]}")
                import traceback
                traceback.print_exc()
                continue

        return results

    def save_results(
        self,
        results: List[Dict[str, Any]],
        config: ChunkingConfig,
    ):
        """Save evaluation results to JSON."""
        output_file = (
            f"results/results_{config.strategy.replace('-', '_')}.json"
        )

        os.makedirs("results", exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to {output_file}")


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

    for config in configs:
        results = evaluator.run_evaluation(config)
        all_results[config.strategy] = results
        evaluator.save_results(results, config)

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
                print(f"  {metric}: N/A (0/{len(results)} valid)")
                continue

            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            print(
                f"  {metric}: {avg_score:.3f} "
                f"(range: {min_score:.3f}-{max_score:.3f}, "
                f"valid: {len(scores)}/{len(results)})"
            )

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()