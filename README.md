# Hybrid RAG System with Advanced Evaluation

A comprehensive Retrieval-Augmented Generation (RAG) system that combines dense vector retrieval, sparse keyword retrieval (BM25), and Reciprocal Rank Fusion (RRF) to answer questions from Wikipedia articles with automated evaluation and innovative metrics.

## 🏗️ Architecture Overview

The system implements three retrieval methods:

1. **Dense Retrieval**: TF-IDF with FAISS indexing for semantic similarity
2. **Sparse Retrieval**: BM25 algorithm with custom preprocessing
3. **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) combining both methods

## 📋 System Requirements

- Python 3.8+
- 4GB+ RAM
- 2GB+ storage space
- Internet connection (for data collection)

## Features

- **Hybrid Retrieval**: Combines dense vector embeddings and sparse BM25 keyword matching
- **Reciprocal Rank Fusion**: Intelligently fuses results from multiple retrieval methods
- **Automated Evaluation**: Comprehensive evaluation framework with 100 generated questions
- **Interactive Interface**: User-friendly Streamlit web application
- **Advanced Metrics**: MRR, ROUGE-L F1, Contextual Precision@10, and innovative metrics
- **Comprehensive Analysis**: Error analysis, ablation studies, and performance visualizations

## Project Structure

```
RagSystem_AI/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── wikipedia_collector.py      # Wikipedia data collection
│   │   ├── text_processor.py           # Text preprocessing and chunking
│   │   └── data_utils.py               # Data utilities
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense_retrieval.py          # Dense vector retrieval
│   │   ├── sparse_retrieval.py         # BM25 sparse retrieval
│   │   └── hybrid_fusion.py            # RRF fusion logic
│   ├── generation/
│   │   ├── __init__.py
│   │   └── response_generator.py       # LLM response generation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── question_generator.py       # Automated question generation
│   │   ├── metrics.py                  # Evaluation metrics
│   │   └── evaluator.py               # Main evaluation pipeline
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py           # Streamlit interface
│   └── main.py                        # Main RAG system
├── data/
│   ├── raw/                          # Raw Wikipedia data
│   ├── processed/                    # Processed chunks
│   ├── indexes/                      # Vector and BM25 indexes
│   └── fixed_urls.json              # Fixed 200 Wikipedia URLs
├── outputs/                          # Evaluation results and reports
├── notebooks/                        # Jupyter notebooks for analysis
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## Installation

1. Clone the repository:
```bash
cd RagSystem_AI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLP models:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
```

## Usage

### 1. Data Collection and Indexing

```bash
# Collect Wikipedia data and build indexes
python src/main.py --mode collect
```

### 2. Run the Streamlit Interface

```bash
streamlit run src/ui/streamlit_app.py
```

### 3. Automated Evaluation

```bash
# Run comprehensive evaluation with 100 questions
python src/main.py --mode evaluate

# Run evaluation with custom parameters
python src/main.py --mode evaluate --output-dir outputs/my_eval

# Full pipeline including evaluation
python src/main.py --mode full
```

The evaluation framework includes:
- **Question Generation**: Template-based and model-based questions from Wikipedia corpus
- **Mean Reciprocal Rank (MRR)**: Measures URL-level retrieval accuracy
- **ROUGE-L**: Evaluates answer quality with longest common subsequence
- **Precision@K**: Assesses retrieval context relevance
- **Semantic Similarity**: Uses sentence embeddings for semantic alignment
- **Ablation Studies**: Compares dense-only, sparse-only, and hybrid approaches
- **Error Analysis**: Categorizes failures and provides improvement recommendations

### 4. Complete Pipeline

```bash
# Run the complete pipeline: collect data, generate questions, and evaluate
python src/main.py --mode full
```

## Configuration

The system can be configured through environment variables or command-line arguments:

- `EMBEDDING_MODEL`: Sentence transformer model (default: 'all-MiniLM-L6-v2')
- `LLM_MODEL`: Language model for generation (default: 'microsoft/DialoGPT-medium')
- `CHUNK_SIZE`: Text chunk size in tokens (default: 300)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `TOP_K_DENSE`: Top-K for dense retrieval (default: 20)
- `TOP_K_SPARSE`: Top-K for sparse retrieval (default: 20)
- `TOP_N_FINAL`: Final number of chunks for generation (default: 10)

## Dataset Requirements

### Fixed URLs (200)
The system maintains a fixed set of 200 Wikipedia URLs stored in `data/fixed_urls.json`. These URLs are unique to each group and remain constant across all indexing operations.

### Random URLs (300)
For each indexing run, the system randomly samples 300 additional Wikipedia URLs (minimum 200 words per page). These change with each rebuild.

### Total Corpus: 500 URLs
- Text extraction and cleaning
- Chunking into 200-400 tokens with 50-token overlap  
- Metadata storage (URL, title, unique chunk IDs)

## Evaluation Framework

### Question Generation
- Automatically generates 100 diverse Q&A pairs from the Wikipedia corpus
- Includes factual, comparative, inferential, and multi-hop questions
- Stores ground truth answers and source information

### Evaluation Metrics

1. **Mean Reciprocal Rank (MRR)** - URL Level (Mandatory)
   - Measures how quickly the system identifies correct source documents
   - Calculated at URL level, not chunk level

2. **Custom Metrics** (2 additional metrics):
   - **ROUGE-L**: Measures answer quality using longest common subsequence
   - **Retrieval Precision@10**: Measures accuracy of top-10 retrieved chunks

3. **Innovative Evaluation Features**:
   - Ablation studies comparing dense-only, sparse-only, and hybrid approaches
   - Error analysis with failure categorization
   - Confidence calibration and correlation analysis
   - Interactive evaluation dashboard

## Architecture

The system implements a hybrid retrieval architecture:

1. **Dense Retrieval**: Uses sentence transformers to encode queries and documents
2. **Sparse Retrieval**: Implements BM25 for keyword-based matching
3. **Reciprocal Rank Fusion**: Combines results using RRF with k=60
4. **Response Generation**: Uses transformer models for final answer generation

## API Documentation

### Core Classes

- `WikipediaCollector`: Handles Wikipedia data collection and processing
- `DenseRetriever`: Implements dense vector retrieval with FAISS
- `SparseRetriever`: Implements BM25 sparse retrieval
- `HybridFusion`: Combines retrieval results using RRF
- `ResponseGenerator`: Generates final answers using LLMs
- `RAGEvaluator`: Comprehensive evaluation framework

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Authors

- Hybrid RAG System Team
- Course: Advanced Information Retrieval
- Date: January 2026