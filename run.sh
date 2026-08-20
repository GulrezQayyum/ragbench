

set -e  # Exit on error

echo "RAGBench Evaluation Runner"
echo "=============================="
echo ""

# Check for GROQ_API_KEY
if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ Error: GROQ_API_KEY not set"
    echo "   Run: export GROQ_API_KEY='your_key'"
    exit 1
fi

echo "GROQ_API_KEY found"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo " Dependencies installed"
echo ""

# Run evaluation
echo "   Running evaluation..."
echo "   Testing semantic chunking..."
echo "   Testing parent-child chunking..."
python eval.py

echo ""
echo "Analyzing results..."
python analyze_results.py

echo ""
echo "COMPLETE!"
echo ""
echo "   Check these files:"
echo "   • results/EVALUATION_REPORT.md (detailed findings)"
echo "   • results/query_analysis.json (per-query comparison)"
echo "   • results/results_semantic.json (raw semantic results)"
echo "   • results/results_parent-child.json (raw parent-child results)"
echo ""
echo "Next: Read results/EVALUATION_REPORT.md for interpretation"