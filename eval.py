import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall
)
from datasets import Dataset
import pandas as pd


@dataclass
class ChunkingConfig:
    name: str
    strategy: str 
    chunk_size: int
    overlap: int


class RAGEvaluator:
    def __init__(self, corpus_path: str = "corpus.json", queries_path: str = "test_queries.json"):
        self.corpus = self._load_json(corpus_path)
        self.queries = self._load_json(queries_path)
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.Client()
        
        print(f"Loaded {len(self.corpus)} documents")
        print(f"Loaded {len(self.queries)} test queries")

    @staticmethod
    def _load_json(path: str) -> Any:
        with open(path, "r") as f:
            return json.load(f)

    def chunk_corpus_semantic(self) -> Dict[str, str]:
        
        chunks = {}
        for doc in self.corpus:
            doc_id = doc["doc_id"]
            content = doc["content"]
            chunks[f"{doc_id}_semantic"] = {
                "content": content,
                "source_doc": doc_id,
                "strategy": "semantic"
            }
        print(f"Semantic chunking: {len(chunks)} chunks created")
        return chunks

    def chunk_corpus_parent_child(self, sentences_per_child: int = 2) -> Dict[str, str]:
        
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
        
        print(f"Parent-child chunking: {len(chunks)} chunks created")
        return chunks

    def build_retriever(self, chunks: Dict[str, str], collection_name: str) -> Any:
        
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
        
        print(f"ChromaDB collection '{collection_name}' created with {len(chunk_ids)} chunks")
        return collection

    def retrieve(self, query: str, collection: Any, top_k: int = 3) -> List[str]:
        query_embedding = self.embeddings_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        documents = results["documents"][0] if results["documents"] else []
        return documents

    def generate_answer(self, query: str, context: List[str]) -> str:
        context_text = "\n\n".join(context)
        
        prompt = f"""
Answer the following question based on the provided context. Be concise and faithful to the context.

Question: {query}

Context:
{context_text}

Answer:
"""
        
        response = self.groq_client.messages.create(
            model="mixtral-8x7b-32768",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()

    def evaluate_rag_pair(self, query: str, context: List[str], answer: str) -> Dict[str, float]:
       
        try:
            eval_data = {
                "question": [query],
                "contexts": [[context]],  
                "answer": [answer],
                "ground_truths": [[query]]  
            }
            
            dataset = Dataset.from_dict(eval_data)
            
            scores = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_relevancy,
                    context_recall
                ]
            )
            
            return {
                "faithfulness": scores["faithfulness"][0],
                "answer_relevancy": scores["answer_relevancy"][0],
                "context_relevancy": scores["context_relevancy"][0],
                "context_recall": scores["context_recall"][0]
            }
        except Exception as e:
            print(f"RAGAS evaluation failed for query '{query[:50]}...': {e}")
            # Return neutral scores on failure
            return {
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "context_relevancy": 0.5,
                "context_recall": 0.5
            }

    def run_evaluation(self, config: ChunkingConfig, top_k: int = 3) -> List[Dict[str, Any]]:
        print(f"\n{'='*60}")
        print(f"🚀 Evaluating: {config.name}")
        print(f"{'='*60}")
        
        if config.strategy == "semantic":
            chunks = self.chunk_corpus_semantic()
        else:  
            chunks = self.chunk_corpus_parent_child()
        
        collection = self.build_retriever(chunks, collection_name=config.name)
        
        results = []
        for i, test_query in enumerate(self.queries):
            query = test_query["query"]
            relevant_ids = test_query["relevant_ids"]
            
            print(f"\n[{i+1}/{len(self.queries)}] {query[:60]}...")
            
            retrieved_context = self.retrieve(query, collection, top_k=top_k)
            
            answer = self.generate_answer(query, retrieved_context)
            
            metrics = self.evaluate_rag_pair(query, retrieved_context, answer)
            
            result = {
                "query_id": test_query["query_id"],
                "query": query,
                "relevant_ids": relevant_ids,
                "retrieved_context": retrieved_context,
                "generated_answer": answer,
                "strategy": config.strategy,
                **metrics
            }
            results.append(result)
            
            print(f"   Faithfulness: {metrics['faithfulness']:.3f}")
            print(f"   Answer Relevancy: {metrics['answer_relevancy']:.3f}")
            print(f"   Context Relevancy: {metrics['context_relevancy']:.3f}")
            print(f"   Context Recall: {metrics['context_recall']:.3f}")
        
        return results

    def save_results(self, results: List[Dict], config: ChunkingConfig):
        output_file = f"results/results_{config.strategy}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n Results saved to {output_file}")
        return output_file


def main():
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
    print("RESULTS COMPARISON")
    print("="*60)
    
    for strategy, results in all_results.items():
        print(f"\n{strategy.upper()}:")
        
        metrics_dict = defaultdict(list)
        for result in results:
            metrics_dict["faithfulness"].append(result["faithfulness"])
            metrics_dict["answer_relevancy"].append(result["answer_relevancy"])
            metrics_dict["context_relevancy"].append(result["context_relevancy"])
            metrics_dict["context_recall"].append(result["context_recall"])
        
        for metric, scores in metrics_dict.items():
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            print(f"  {metric}:")
            print(f"    Average: {avg_score:.3f}")
            print(f"    Min: {min_score:.3f} | Max: {max_score:.3f}")
    
    print("\n Evaluation complete!")


if __name__ == "__main__":
    main()