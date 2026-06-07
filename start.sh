#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AI Resume Screener — Quick Start
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "🚀  AI Resume Screener"
echo "────────────────────────────────────────────"

# 1. Install dependencies
echo "📦  Installing dependencies..."
pip install -r requirements.txt --break-system-packages -q

# 2. Download NLTK data
echo "📚  Downloading NLTK data..."
python3 -c "
import nltk
for pkg in ('punkt', 'stopwords', 'punkt_tab'):
    nltk.download(pkg, quiet=True)
print('  ✓ NLTK ready')
"

# 3. Start Flask
echo ""
echo "✅  Starting server at http://localhost:5050"
echo "   Open your browser → http://localhost:5050"
echo "────────────────────────────────────────────"
echo ""

python3 app.py
