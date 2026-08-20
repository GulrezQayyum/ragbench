import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics


class ResultsAnalyzer:
    def __init__(self):
        self.semantic_results = None
        self.parent_child_results = None

    def load_results(self, semantic_path: str = "results/results_semantic.json",
                     parent_child_path: str = "results/results_parent-child.json"):
        with open(semantic_path, "r") as f:
            self.semantic_results = json.load(f)
        
        with open(parent_child_path, "r") as f:
            self.parent_child_results = json.load(f)
        
        print(f"Loaded {len(self.semantic_results)} semantic results")
        print(f"Loaded {len(self.parent_child_results)} parent-child results")

    def aggregate_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        metrics = ["faithfulness", "answer_relevancy", "context_relevancy", "context_recall"]
        aggregated = {}
        
        for metric in metrics:
            scores = [r[metric] for r in results]
            aggregated[metric] = {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores),
                "scores": scores
            }
        
        return aggregated

    def compare_strategies(self) -> Dict[str, any]:
        semantic_agg = self.aggregate_metrics(self.semantic_results)
        parent_child_agg = self.aggregate_metrics(self.parent_child_results)
        
        comparison = {}
        
        for metric in semantic_agg.keys():
            sem_mean = semantic_agg[metric]["mean"]
            pc_mean = parent_child_agg[metric]["mean"]
            
            diff = pc_mean - sem_mean
            diff_pct = (diff / sem_mean * 100) if sem_mean != 0 else 0
            
            comparison[metric] = {
                "semantic": sem_mean,
                "parent_child": pc_mean,
                "difference": diff,
                "difference_pct": diff_pct,
                "winner": "parent-child" if diff > 0 else "semantic",
                "semantic_stdev": semantic_agg[metric]["stdev"],
                "parent_child_stdev": parent_child_agg[metric]["stdev"]
            }
        
        return comparison

    def query_level_analysis(self) -> List[Dict]:
        analysis = []
        
        for sem_result, pc_result in zip(self.semantic_results, self.parent_child_results):
            query = sem_result["query"]
            
            sem_avg = sum([
                sem_result["faithfulness"],
                sem_result["answer_relevancy"],
                sem_result["context_relevancy"],
                sem_result["context_recall"]
            ]) / 4
            
            pc_avg = sum([
                pc_result["faithfulness"],
                pc_result["answer_relevancy"],
                pc_result["context_relevancy"],
                pc_result["context_recall"]
            ]) / 4
            
            winner = "parent-child" if pc_avg > sem_avg else "semantic"
            margin = abs(pc_avg - sem_avg)
            
            analysis.append({
                "query": query,
                "semantic_score": sem_avg,
                "parent_child_score": pc_avg,
                "winner": winner,
                "margin": margin,
                "semantic_details": {
                    "faithfulness": sem_result["faithfulness"],
                    "answer_relevancy": sem_result["answer_relevancy"],
                    "context_relevancy": sem_result["context_relevancy"],
                    "context_recall": sem_result["context_recall"]
                },
                "parent_child_details": {
                    "faithfulness": pc_result["faithfulness"],
                    "answer_relevancy": pc_result["answer_relevancy"],
                    "context_relevancy": pc_result["context_relevancy"],
                    "context_recall": pc_result["context_recall"]
                }
            })
        
        return analysis

    def generate_markdown_report(self) -> str:
        comparison = self.compare_strategies()
        query_analysis = self.query_level_analysis()
        
        sem_wins = sum(1 for qa in query_analysis if qa["winner"] == "semantic")
        pc_wins = sum(1 for qa in query_analysis if qa["winner"] == "parent-child")
        
        report = f"""# RAGBench Evaluation Report

## Executive Summary

This report compares **semantic chunking** vs **parent-child chunking** strategies for RAG systems.

**Query-level Results:**
- Semantic Chunking wins: {sem_wins}/{len(query_analysis)} ({100*sem_wins/len(query_analysis):.1f}%)
- Parent-Child Chunking wins: {pc_wins}/{len(query_analysis)} ({100*pc_wins/len(query_analysis):.1f}%)

---

## Metric-Level Comparison

### Overall Scores

| Metric | Semantic | Parent-Child | Difference | Winner |
|--------|----------|--------------|-----------|--------|
"""
        
        for metric, comp in comparison.items():
            diff = comp["difference"]
            diff_pct = comp["difference_pct"]
            winner = comp["winner"].upper()
            report += f"| {metric} | {comp['semantic']:.3f} | {comp['parent_child']:.3f} | {diff:+.3f} ({diff_pct:+.1f}%) | {winner} |\n"
        
        report += f"""

### Detailed Metric Analysis

"""
        
        for metric, comp in comparison.items():
            report += f"""#### {metric.replace('_', ' ').title()}

**Semantic Chunking:**
- Mean: {comp['semantic']:.3f}
- Std Dev: {comp['semantic_stdev']:.3f}

**Parent-Child Chunking:**
- Mean: {comp['parent_child']:.3f}
- Std Dev: {comp['parent_child_stdev']:.3f}

**Interpretation:**
"""
            
            if metric == "faithfulness":
                if comp["winner"] == "semantic":
                    report += f"Semantic chunking is more faithful ({comp['semantic']:.3f} vs {comp['parent_child']:.3f}). Smaller, topically-coherent chunks reduce hallucination.\n"
                else:
                    report += f"Parent-child is more faithful ({comp['parent_child']:.3f} vs {comp['semantic']:.3f}). Larger parent context provides grounding.\n"
            
            elif metric == "context_relevancy":
                if comp["winner"] == "semantic":
                    report += f"Semantic chunks are more relevant ({comp['semantic']:.3f}). Topic boundaries align well with query intent.\n"
                else:
                    report += f"Parent-child retrieves more relevant context ({comp['parent_child']:.3f}). Child + parent ensures nothing is missed.\n"
            
            elif metric == "answer_relevancy":
                if comp["winner"] == "semantic":
                    report += f"Semantic answers are more on-topic ({comp['semantic']:.3f}). Precise chunks reduce noise.\n"
                else:
                    report += f"Parent-child answers are more relevant ({comp['parent_child']:.3f}). Broader context helps answer specificity.\n"
            
            elif metric == "context_recall":
                if comp["winner"] == "semantic":
                    report += f"Semantic chunking has higher recall ({comp['semantic']:.3f}). Each topic stays intact.\n"
                else:
                    report += f"Parent-child has higher recall ({comp['parent_child']:.3f}). Parent chunks guarantee full document coverage.\n"
            
            report += "\n"
        
        report += """
---

## Query-Level Results

Showing queries where one strategy significantly outperformed the other:

"""
        
        semantic_favored = sorted([q for q in query_analysis if q["winner"] == "semantic"], 
                                 key=lambda x: x["margin"], reverse=True)[:5]
        pc_favored = sorted([q for q in query_analysis if q["winner"] == "parent-child"], 
                           key=lambda x: x["margin"], reverse=True)[:5]
        
        report += "### Queries Favoring Semantic Chunking\n\n"
        for qa in semantic_favored:
            report += f"""
**Q: {qa['query']}**
- Semantic: {qa['semantic_score']:.3f}
- Parent-Child: {qa['parent_child_score']:.3f}
- Margin: {qa['margin']:+.3f}
"""
        
        report += "\n### Queries Favoring Parent-Child Chunking\n\n"
        for qa in pc_favored:
            report += f"""
**Q: {qa['query']}**
- Parent-Child: {qa['parent_child_score']:.3f}
- Semantic: {qa['semantic_score']:.3f}
- Margin: {qa['margin']:+.3f}
"""
        
        report += f"""

---

## Recommendations

Based on the evaluation:

1. **For Precision-Critical Tasks** (e.g., Q&A with hallucination sensitivity):
   - Use Semantic Chunking if faithfulness > {comparison['faithfulness']['semantic']:.3f}
   - Or Parent-Child if you need broader context

2. **For Coverage-Critical Tasks** (e.g., multi-hop reasoning):
   - Parent-Child likely performs better (context_recall: {comparison['context_recall']['parent_child']:.3f})

3. **Hybrid Approach**:
   - Use semantic chunking as primary retriever
   - Fall back to parent chunks if answer confidence is low

---

## Limitations

- Evaluation limited to {len(query_analysis)} test queries
- Ground truth limited to expected document references (not full answer correctness)
- RAGAS metrics are LLM-based; scores depend on scoring model used
- No human evaluation; metrics are proxy measures

"""
        
        return report

    def save_report(self, report: str):
        os.makedirs("results", exist_ok=True)
        
        with open("results/EVALUATION_REPORT.md", "w") as f:
            f.write(report)
        
        print("Report saved to results/EVALUATION_REPORT.md")

    def run_analysis(self):
        print("\n" + "="*60)
        print("ANALYZING RESULTS")
        print("="*60 + "\n")
        
        comparison = self.compare_strategies()
        query_analysis = self.query_level_analysis()
        
        print("Metric Comparison (Semantic vs Parent-Child):\n")
        for metric, comp in comparison.items():
            print(f"{metric}:")
            print(f"  Semantic: {comp['semantic']:.3f}")
            print(f"  Parent-Child: {comp['parent_child']:.3f}")
            print(f"  Winner: {comp['winner']} ({comp['difference_pct']:+.1f}%)\n")
        
        report = self.generate_markdown_report()
        self.save_report(report)
        
        os.makedirs("results", exist_ok=True)
        with open("results/query_analysis.json", "w") as f:
            json.dump(query_analysis, f, indent=2)
        
        print("\nAnalysis complete!")
        print("Check results/EVALUATION_REPORT.md for detailed findings")


def main():
    analyzer = ResultsAnalyzer()
    
    try:
        analyzer.load_results()
        analyzer.run_analysis()
    except FileNotFoundError as e:
        print(f" Error: {e}")
        print(" Make sure to run eval.py first to generate results files")


if __name__ == "__main__":
    main()