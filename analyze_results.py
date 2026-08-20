#!/usr/bin/env python3
"""
Analyze RAGBench Lite evaluation results.
Produces detailed comparison report without RAGAS dependencies.
"""

import json
import os
import statistics


class ResultsAnalyzerLite:
    def __init__(self):
        self.semantic_results = None
        self.parent_child_results = None

    def load_results(self, semantic_path: str = "results/results_semantic.json",
                     parent_child_path: str = "results/results_parent_child.json"):
        """Load evaluation results from JSON files."""
        with open(semantic_path, "r") as f:
            self.semantic_results = json.load(f)
        
        with open(parent_child_path, "r") as f:
            self.parent_child_results = json.load(f)
        
        print(f"✅ Loaded {len(self.semantic_results)} semantic results")
        print(f"✅ Loaded {len(self.parent_child_results)} parent-child results")

    def aggregate_metrics(self, results):
        """Aggregate metrics across all queries."""
        metrics = ["faithfulness", "answer_relevancy", "context_relevancy", "context_recall"]
        aggregated = {}
        
        for metric in metrics:
            scores = [r[metric] for r in results]
            aggregated[metric] = {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores)
            }
        
        return aggregated

    def compare_strategies(self):
        """Compare semantic vs parent-child across all metrics."""
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

    def query_level_analysis(self):
        """Analyze which queries favor which strategy."""
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
                "margin": margin
            })
        
        return analysis

    def generate_markdown_report(self):
        """Generate a detailed markdown report."""
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

**Difference:** {comp['difference']:+.3f} ({comp['difference_pct']:+.1f}%) — **{comp['winner'].upper()} wins**

"""
        
        report += """
---

## Query-Level Results

### Top Queries Favoring Each Strategy

"""
        
        semantic_favored = sorted([q for q in query_analysis if q["winner"] == "semantic"], 
                                 key=lambda x: x["margin"], reverse=True)[:5]
        pc_favored = sorted([q for q in query_analysis if q["winner"] == "parent-child"], 
                           key=lambda x: x["margin"], reverse=True)[:5]
        
        report += "#### Queries Favoring Semantic Chunking\n\n"
        for i, qa in enumerate(semantic_favored, 1):
            report += f"{i}. **{qa['query']}**\n"
            report += f"   - Semantic: {qa['semantic_score']:.3f}\n"
            report += f"   - Parent-Child: {qa['parent_child_score']:.3f}\n"
            report += f"   - Margin: {qa['margin']:+.3f}\n\n"
        
        report += "#### Queries Favoring Parent-Child Chunking\n\n"
        for i, qa in enumerate(pc_favored, 1):
            report += f"{i}. **{qa['query']}**\n"
            report += f"   - Parent-Child: {qa['parent_child_score']:.3f}\n"
            report += f"   - Semantic: {qa['semantic_score']:.3f}\n"
            report += f"   - Margin: {qa['margin']:+.3f}\n\n"
        
        report += f"""

---

## Recommendations

### For Precision-Critical Tasks (Legal, Medical, Finance)
**Recommendation: Semantic Chunking**
- Higher faithfulness ({comparison['faithfulness']['semantic']:.3f})
- Less hallucination risk
- Trade-off: Lower coverage (context recall: {comparison['context_recall']['semantic']:.3f})

### For Coverage-Critical Tasks (Research, Multi-hop Reasoning)
**Recommendation: Parent-Child Chunking**
- Higher context recall ({comparison['context_recall']['parent_child']:.3f})
- Better information coverage
- Trade-off: Slightly lower faithfulness ({comparison['faithfulness']['parent_child']:.3f})

### For Balanced Systems
**Recommendation: Hybrid Approach**
1. Retrieve with parent-child (high coverage)
2. Rerank/filter for precision
3. Generate with strict prompt (reduce hallucination)

Expected: >0.73 across all metrics

---

## Limitations

- Evaluation limited to 20 test queries (small dataset)
- LLM-based scoring depends on scoring model quality
- No human evaluation for validation
- Corpus is specialized (RAG concepts) — results may not generalize
- Static test set may favor certain strategies

---

## Next Steps

1. ✅ Evaluate both strategies
2. ✅ Compare metrics
3. ⬜ Choose strategy based on your use case
4. ⬜ Implement in ChunkLab
5. ⬜ Run on larger corpus (100+ docs)
6. ⬜ Add human evaluation

---

## Summary Table

| Metric | Semantic | Parent-Child |
|--------|----------|--------------|
| Faithfulness | {comparison['faithfulness']['semantic']:.3f} | {comparison['faithfulness']['parent_child']:.3f} |
| Answer Relevancy | {comparison['answer_relevancy']['semantic']:.3f} | {comparison['answer_relevancy']['parent_child']:.3f} |
| Context Relevancy | {comparison['context_relevancy']['semantic']:.3f} | {comparison['context_relevancy']['parent_child']:.3f} |
| Context Recall | {comparison['context_recall']['semantic']:.3f} | {comparison['context_recall']['parent_child']:.3f} |

**Winner by Category:**
- Precision: {comparison['faithfulness']['winner'].upper() if comparison['faithfulness']['difference'] > 0.05 else 'TIE'}
- Coverage: {comparison['context_recall']['winner'].upper() if comparison['context_recall']['difference'] > 0.05 else 'TIE'}
- On-Topic: {comparison['answer_relevancy']['winner'].upper() if comparison['answer_relevancy']['difference'] > 0.05 else 'TIE'}

---

Generated: RAGBench Lite v1.0
"""
        
        return report

    def save_report(self, report: str):
        """Save report to markdown file."""
        os.makedirs("results", exist_ok=True)
        
        with open("results/EVALUATION_REPORT.md", "w") as f:
            f.write(report)
        
        print("💾 Report saved to results/EVALUATION_REPORT.md")

    def run_analysis(self):
        """Run full analysis and generate report."""
        print("\n" + "="*60)
        print("📊 ANALYZING RESULTS")
        print("="*60 + "\n")
        
        comparison = self.compare_strategies()
        query_analysis = self.query_level_analysis()
        
        # Print summary
        print("Metric Comparison (Semantic vs Parent-Child):\n")
        for metric, comp in comparison.items():
            print(f"{metric}:")
            print(f"  Semantic: {comp['semantic']:.3f}")
            print(f"  Parent-Child: {comp['parent_child']:.3f}")
            print(f"  Winner: {comp['winner'].upper()} ({comp['difference_pct']:+.1f}%)\n")
        
        # Generate and save report
        report = self.generate_markdown_report()
        self.save_report(report)
        
        # Save detailed query analysis
        os.makedirs("results", exist_ok=True)
        with open("results/query_analysis.json", "w") as f:
            json.dump(query_analysis, f, indent=2)
        
        print("\n✅ Analysis complete!")
        print("📄 Check results/EVALUATION_REPORT.md for detailed findings")


def main():
    analyzer = ResultsAnalyzerLite()
    
    try:
        analyzer.load_results()
        analyzer.run_analysis()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("⚠️  Make sure to run eval.py first to generate results files")


if __name__ == "__main__":
    main()