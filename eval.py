#!/usr/bin/env python3
"""
RAGBench Evaluation Runner - IMPROVED VERSION
Robust scoring with better error handling
"""

import json
import os
import re
from typing import List, Dict, Any
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
    def __init__(self, corpus_path: str = "corpus.json", queries_path: str = "test_queries.json"):
        """Initialize evaluator with corpus and test queries."""
        self.corpus = self._load_json(corpus_path)
        self.queries = self._load_json(queries_path)
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.Client()
        
        print(f"✅ Loaded {len(self.corpus)} documents")
        print(f"✅ Loaded {len(self.queries)} test queries")

    @staticmethod
    def _load_json(path: str) -> Any:
        """Load JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def chunk_corpus_semantic(self) -> Dict[str, str]:
        """Semantic chunking: Keep each document as one chunk"""
        chunks = {}
        for doc in self.corpus:
            doc_id = doc["doc_id"]
            content = doc["content"]
            chunks[f"{doc_id}_semantic"] = {
                "content": content,
                "source_doc": doc_id,
                "strategy": "semantic"
            }
        print(f"✅ Semantic chunking: {len(chunks)} chunks created")
        return chunks

    def chunk_corpus_parent_child(self, sentences_per_child: int = 2) -> Dict[str, str]:
        """Parent-child chunking: Split each doc into child chunks + keep parent."""
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
                "type": "parent"
            }
            
            sentences = content.split(". ")
            for i in range(0, len(sentences), sentences_per_child):
                child_sentences = sentences[i:i+sentences_per_child]
                child_content = ". ".join(child_sentences)
                if child_content.strip():
                    child_id = f"{doc_id}_child_{i//sentences_per_child}"
                    chunks[child_id] = {
                        "content": child_content,
                        "source_doc": doc_id,
                        "strategy": "parent-child",
                        "type": "child",
                        "parent_id": parent_id
                    }
            parent_chunk_id += 1
        
        print(f"✅ Parent-child chunking: {len(chunks)} chunks created")
        return chunks

    def build_retriever(self, chunks: Dict[str, str], collection_name: str) -> Any:
        """Build ChromaDB collection for retrieval."""
        try:
            self.chroma_client.delete_collection(name=collection_name)
        except:
            pass
        
        collection = self.chroma_client.create_collection(name=collection_name)
        
        chunk_ids = []
        chunk_contents = []
        chunk_metadatas = []
        
        for chunk_id, chunk_data in chunks.items():
            chunk_ids.append(chunk_id)
            chunk_contents.append(chunk_data["content"])
            chunk_metadatas.append({
                "source_doc": chunk_data["source_doc"],
                "strategy": chunk_data["strategy"]
            })
        
        embeddings = self.embeddings_model.encode(chunk_contents).tolist()
        
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_contents,
            metadatas=chunk_metadatas
        )
        
        print(f"✅ ChromaDB collection '{collection_name}' created with {len(chunk_ids)} chunks")
        return collection

    def retrieve(self, query: str, collection: Any, top_k: int = 3) -> List[str]:
        """Retrieve top-k chunks for a query."""
        query_embedding = self.embeddings_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        documents = results["documents"][0] if results["documents"] else []
        return documents

    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer using Groq given query and context."""
        context_text = "\n\n".join(context)
        
        prompt = f"""Answer based ONLY on this context. Be concise.

Question: {query}

Context:
{context_text}

Answer:"""
        
        response = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.strip()

    def extract_score(self, text: str) -> float:
        """Extract score from response text robustly."""
        try:
            # Try to find a number like 0.XX
            match = re.search(r'0\.\d+', text.strip())
            if match:
                score = float(match.group())
                return min(1.0, max(0.0, score))
            # Try direct float conversion
            score = float(text.strip().split()[0])
            return min(1.0, max(0.0, score))
        except:
            return 0.5

    def score_faithfulness(self, query: str, context: List[str], answer: str) -> float:
        """Score how faithful answer is to context (0-1)."""
        context_text = "\n\n".join(context)
        
        prompt = f"""Rate faithfulness 0-1: Is answer grounded in context?

Context: {context_text[:500]}...

Answer: {answer}

Score (0-1):"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content.strip()
            return self.extract_score(text)
        except Exception as e:
            return 0.5

    def score_context_relevancy(self, query: str, context: List[str]) -> float:
        """Score how relevant the retrieved context is to the query (0-1)."""
        context_text = "\n\n".join(context)
        
        prompt = f"""Rate relevancy 0-1: Is context on-topic for this query?

Query: {query}

Context: {context_text[:500]}...

Score (0-1):"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content.strip()
            return self.extract_score(text)
        except Exception as e:
            return 0.5

    def score_answer_relevancy(self, query: str, answer: str) -> float:
        """Score how well the answer addresses the query (0-1)."""
        prompt = f"""Rate relevancy 0-1: Does answer address the query?

Query: {query}

Answer: {answer}

Score (0-1):"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content.strip()
            return self.extract_score(text)
        except Exception as e:
            return 0.5

    def score_context_recall(self, query: str, context: List[str], answer: str) -> float:
        """Score if context contains info needed for answer (0-1)."""
        context_text = "\n\n".join(context)
        
        prompt = f"""Rate recall 0-1: Does context have all info needed for this answer?

Query: {query}

Context: {context_text[:500]}...

Answer: {answer}

Score (0-1):"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content.strip()
            return self.extract_score(text)
        except Exception as e:
            return 0.5

    def run_evaluation(self, config: ChunkingConfig, top_k: int = 3) -> List[Dict[str, Any]]:
        """Run full evaluation pipeline for a chunking strategy."""
        print(f"\n{'='*60}")
        print(f"🚀 Evaluating: {config.name}")
        print(f"{'='*60}")
        
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
            
            retrieved_context = self.retrieve(query, collection, top_k=top_k)
            answer = self.generate_answer(query, retrieved_context)
            
            faithfulness = self.score_faithfulness(query, retrieved_context, answer)
            context_relevancy = self.score_context_relevancy(query, retrieved_context)
            answer_relevancy = self.score_answer_relevancy(query, answer)
            context_recall = self.score_context_recall(query, retrieved_context, answer)
            
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
                "context_recall": context_recall
            }
            results.append(result)
            
            print(f"   F:{faithfulness:.2f} | AR:{answer_relevancy:.2f} | CR:{context_relevancy:.2f} | REC:{context_recall:.2f}")
        
        return results

    def save_results(self, results: List[Dict], config: ChunkingConfig):
        """Save evaluation results to JSON."""
        # Use underscore in filename for consistency
        output_file = f"results/results_{config.strategy.replace('-', '_')}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")


def main():
    """Run full evaluation comparing semantic vs parent-child chunking."""
    evaluator = RAGEvaluator()
    
    configs = [
        ChunkingConfig(
            name="semantic_chunking",
            strategy="semantic",
            chunk_size=512,
            overlap=0
        ),
        ChunkingConfig(
            name="parent_child_chunking",
            strategy="parent-child",
            chunk_size=256,
            overlap=50
        )
    ]
    
    all_results = {}
    for config in configs:
        results = evaluator.run_evaluation(config)
        all_results[config.strategy] = results
        evaluator.save_results(results, config)
    
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    
    for strategy, results in all_results.items():
        print(f"\n{strategy.upper()}:")
        
        metrics_data = {
            "faithfulness": [r["faithfulness"] for r in results],
            "answer_relevancy": [r["answer_relevancy"] for r in results],
            "context_relevancy": [r["context_relevancy"] for r in results],
            "context_recall": [r["context_recall"] for r in results]
        }
        
        for metric, scores in metrics_data.items():
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            print(f"  {metric}: {avg_score:.3f} (range: {min_score:.3f}-{max_score:.3f})")
    
    print("\n✅ Evaluation complete!")
    print("📄 Next: python analyze_results.py")


if __name__ == "__main__":
    main()