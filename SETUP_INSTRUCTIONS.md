# Streamlit RAG Interface - Setup Instructions

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB recommended
- **Storage**: At least 2GB free space
- **Internet**: Required for initial data collection and model downloads

## 🚀 Quick Start Guide

### Step 1: Clone or Navigate to Project Directory

```bash
cd /Users/sreeram_gv@optum.com/Library/CloudStorage/OneDrive-UHG/Desktop/RagSystem_AI
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Note**: Installation may take 5-15 minutes depending on your internet connection.

### Step 4: Download Required NLP Models

```bash
# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 5: Set Up Data (First Time Only)

If you don't have data collected yet, run:

```bash
# Collect Wikipedia data
python collect_data.py

# Or use the setup script
bash setup.sh
```

This will:
- Collect Wikipedia articles
- Process and chunk the text
- Create necessary data directories

### Step 6: Launch the Streamlit App

```bash
streamlit run hybrid_rag_streamlit_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 🎯 Using the Application

### Main Features

1. **Query Interface**
   - Enter your question in the text input
   - Select retrieval method (Hybrid, Dense, Sparse, RRF)
   - Adjust number of retrieved documents (k)
   - View results with relevance scores

2. **Document Management**
   - View collection statistics
   - Browse indexed documents
   - See document distribution

3. **Evaluation Dashboard**
   - Run automated evaluations
   - View performance metrics
   - Compare retrieval methods
   - Analyze error patterns

4. **Advanced Settings**
   - Configure fusion parameters
   - Adjust retrieval weights
   - Customize preprocessing options

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
ModuleNotFoundError: No module named 'streamlit'
```

**Solution**: Ensure virtual environment is activated and dependencies are installed

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. NLTK Data Missing

```bash
LookupError: Resource 'punkt' not found
```

**Solution**: Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

#### 3. SpaCy Model Missing

```bash
OSError: [E050] Can't find model 'en_core_web_sm'
```

**Solution**: Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

#### 4. Port Already in Use

```bash
OSError: [Errno 48] Address already in use
```

**Solution**: Use a different port

```bash
streamlit run hybrid_rag_streamlit_app.py --server.port 8502
```

#### 5. No Data Found

```bash
FileNotFoundError: Data directory not found
```

**Solution**: Run data collection

```bash
python collect_data.py
```

### Memory Issues

If you encounter memory errors:

1. Reduce the number of documents indexed
2. Lower the chunk size in preprocessing
3. Use `faiss-cpu` instead of `faiss-gpu` (already in requirements)

## 📦 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | >=1.28.0 | Web interface |
| sentence-transformers | >=2.2.2 | Embeddings |
| faiss-cpu | >=1.8.0 | Vector search |
| rank-bm25 | >=0.2.2 | Sparse retrieval |
| transformers | >=4.30.0 | NLP models |
| torch | >=2.0.0 | Deep learning |
| nltk | >=3.8.0 | Text processing |
| spacy | >=3.7.0 | NLP pipeline |

## 🔧 Configuration Options

### Environment Variables (Optional)

Create a `.env` file in the project root:

```bash
# OpenAI API (if using GPT models)
OPENAI_API_KEY=your_api_key_here

# Hugging Face (for model downloads)
HUGGINGFACE_TOKEN=your_token_here

# Custom settings
RAG_DATA_PATH=./data
RAG_OUTPUT_PATH=./outputs
```

### Streamlit Configuration

Create `.streamlit/config.toml` for custom settings:

```toml
[server]
port = 8501
headless = true
enableCORS = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 🧪 Verifying Installation

Run this verification script:

```bash
python -c "
import streamlit
import sentence_transformers
import faiss
import nltk
import spacy
import torch
print('✅ All core dependencies installed successfully!')
print(f'Streamlit: {streamlit.__version__}')
print(f'PyTorch: {torch.__version__}')
print('🚀 Ready to launch!')
"
```

## 📊 System Test

Test the complete system:

```bash
python test_complete_system.py
```

This will verify:
- Data loading
- Indexing
- Retrieval methods
- Fusion logic
- Evaluation metrics

## 🔄 Updating the System

To update dependencies:

```bash
# Pull latest changes (if using git)
git pull

# Update packages
pip install -r requirements.txt --upgrade

# Relaunch app
streamlit run hybrid_rag_streamlit_app.py
```

## 📝 Alternative Launch Methods

### Method 1: Using Python directly

```bash
python -m streamlit run hybrid_rag_streamlit_app.py
```

### Method 2: Using setup script

```bash
bash setup.sh
```

### Method 3: Custom configuration

```bash
streamlit run hybrid_rag_streamlit_app.py \
  --server.port 8501 \
  --server.address localhost \
  --browser.serverAddress localhost
```

## 🎨 Features Available in the UI

### 1. Home Dashboard
- System overview
- Quick stats
- Recent queries

### 2. Query Interface
- Natural language questions
- Real-time search
- Confidence scores
- Source documents

### 3. Batch Evaluation
- 100+ test questions
- Automated metrics
- Performance comparison
- Visualizations

### 4. Analytics
- Query patterns
- Response times
- Accuracy trends
- Error analysis

### 5. System Monitoring
- Cache statistics
- Index health
- Memory usage
- Performance metrics

## 🔐 Security Notes

- The app runs locally by default
- No data is sent to external servers (except for model downloads)
- Wikipedia data is cached locally
- All processing happens on your machine

## 💡 Performance Tips

1. **First Run**: Initial startup may be slow due to model loading
2. **Caching**: Subsequent queries are faster due to caching
3. **Batch Processing**: Use batch evaluation for multiple questions
4. **Resource Management**: Close unused tabs to free memory

## 📧 Support

If you encounter issues:

1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure sufficient disk space and memory
4. Check console output for error messages

## 🎓 Next Steps

After successful setup:

1. Try sample queries in the interface
2. Explore different retrieval methods
3. Run evaluation benchmarks
4. Customize parameters for your use case
5. Analyze performance metrics

---

**Created**: February 8, 2026  
**Version**: 1.0  
**Status**: Ready for Production Use ✅
