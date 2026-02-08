#!/bin/bash

# Hybrid RAG System Startup Script
# This script helps you get started with the RAG system

echo "🔍 Hybrid RAG System Startup"
echo "==============================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📖 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
echo "🧠 Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Quick Start Options:"
echo ""
echo "1. Run complete pipeline:"
echo "   python src/main.py --mode full"
echo ""
echo "2. Start web interface:"
echo "   streamlit run src/ui/streamlit_app.py"
echo ""
echo "3. Run single query:"
echo "   python src/main.py --mode query --query 'What is AI?'"
echo ""
echo "4. Just collect data:"
echo "   python src/main.py --mode collect"
echo ""
echo "5. Just build indexes:"
echo "   python src/main.py --mode index"
echo ""
echo "📋 For more options: python src/main.py --help"