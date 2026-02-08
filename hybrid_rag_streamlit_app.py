"""
Modern Hybrid RAG System Streamlit Interface
Complete interface with advanced features and professional UI
"""

import streamlit as st
import json
import sys
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import RAG system components
try:
    from src.main import HybridRAGSystem
    from src.data.text_processor import TextProcessor
    from src.evaluation.advanced_evaluator import AdvancedEvaluationSuite
    from src.evaluation.evaluation_visualizers import EvaluationVisualizer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Try importing advanced evaluation dependencies
try:
    from sentence_transformers import SentenceTransformer
    ADVANCED_EVALUATION_AVAILABLE = True
except ImportError:
    ADVANCED_EVALUATION_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Hybrid RAG System - Advanced Information Retrieval"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    
    .score-badge {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .stApp {
        background-color: #f5f6fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_system_data():
    """Load all system data with caching"""
    try:
        data_dir = PROJECT_ROOT / "data"
        
        # Load processed chunks
        chunks_path = data_dir / "processed" / "chunks.json"
        metadata_path = data_dir / "processed" / "metadata.json"
        config_path = data_dir / "config.json"
        stats_path = data_dir / "system_stats.json"
        
        chunks = []
        metadata = {}
        config = {}
        stats = {}
        
        if chunks_path.exists():
            with open(chunks_path, 'r') as f:
                chunks = json.load(f)
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        
        return {
            'chunks': chunks,
            'metadata': metadata,
            'config': config,
            'stats': stats,
            'status': 'loaded' if chunks else 'no_data'
        }
        
    except Exception as e:
        return {
            'chunks': [],
            'metadata': {},
            'config': {},
            'stats': {},
            'status': f'error: {e}'
        }

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching"""
    if not RAG_AVAILABLE:
        return None
    
    try:
        rag = HybridRAGSystem()
        return rag
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

def perform_hybrid_search(query: str, chunks: List[Dict], top_k: int = 10) -> List[Dict]:
    """Perform hybrid search using multiple methods"""
    if not chunks or not query:
        return []
    
    results = []
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    for i, chunk in enumerate(chunks):
        content = chunk.get('content', '').lower()
        title = chunk.get('article_title', '').lower()
        
        # Multiple scoring methods
        scores = {}
        
        # 1. Keyword overlap score
        content_words = set(content.split())
        title_words = set(title.split())
        
        content_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
        title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
        scores['keyword'] = content_overlap + (title_overlap * 2)
        
        # 2. Substring matching
        content_matches = sum(content.count(word) for word in query_words)
        title_matches = sum(title.count(word) for word in query_words)
        scores['substring'] = content_matches + (title_matches * 2)
        
        # 3. Position-based scoring (earlier matches get higher scores)
        position_score = 0
        for word in query_words:
            pos = content.find(word)
            if pos != -1:
                position_score += max(0, 1 - (pos / len(content)))
        scores['position'] = position_score
        
        # 4. Length normalization
        content_length = len(chunk.get('content', ''))
        scores['length_norm'] = 1 / (1 + content_length / 1000)  # Prefer shorter, focused chunks
        
        # Combined score
        final_score = (
            scores['keyword'] * 0.4 +
            scores['substring'] * 0.3 +
            scores['position'] * 0.2 +
            scores['length_norm'] * 0.1
        )
        
        if final_score > 0:
            results.append({
                'chunk_id': chunk.get('chunk_id', i),
                'content': chunk.get('content', ''),
                'article_title': chunk.get('article_title', 'Unknown'),
                'article_url': chunk.get('article_url', ''),
                'score': final_score,
                'scores_breakdown': scores,
                'chunk_index': i
            })
    
    # Sort by score and return top_k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def display_search_results(results: List[Dict], query: str):
    """Display search results with rich formatting"""
    if not results:
        st.warning("🤔 No results found. Try different keywords or check your spelling.")
        return
    
    st.markdown(f"### 🎯 Found {len(results)} relevant results")
    
    for i, result in enumerate(results, 1):
        with st.container():
            col1, col2 = st.columns([0.7, 0.3])
            
            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <h4 style="color: #2c3e50; margin-bottom: 0.5rem;">
                        {i}. {result['article_title']}
                    </h4>
                    <div style="margin-bottom: 1rem;">
                        <span class="score-badge">Score: {result['score']:.3f}</span>
                    </div>
                    <p style="color: #34495e; line-height: 1.6;">
                        {result['content'][:500]}{'...' if len(result['content']) > 500 else ''}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Score breakdown
                scores = result.get('scores_breakdown', {})
                
                if scores:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(scores.values()),
                            y=list(scores.keys()),
                            orientation='h',
                            marker_color='rgba(102, 126, 234, 0.8)'
                        )
                    ])
                    
                    fig.update_layout(
                        title=f"Score Breakdown",
                        height=200,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, width='stretch')
            
            # Article link
            if result.get('article_url'):
                st.markdown(f"🔗 [Read full article]({result['article_url']})")
            
            st.markdown("---")

def create_dashboard(data: Dict):
    """Create system dashboard"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Hybrid RAG System Dashboard</h1>
        <p>Advanced Information Retrieval with Semantic and Keyword Search</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        chunk_count = len(data.get('chunks', []))
        st.metric(
            label="📚 Total Chunks",
            value=f"{chunk_count:,}",
            delta=f"Ready for search" if chunk_count > 0 else "No data"
        )
    
    with col2:
        articles = set(chunk.get('article_title', '') for chunk in data.get('chunks', []))
        article_count = len(articles) if articles else 0
        st.metric(
            label="📖 Unique Articles",
            value=f"{article_count:,}",
            delta="Knowledge sources"
        )
    
    with col3:
        total_words = sum(len(chunk.get('content', '').split()) for chunk in data.get('chunks', []))
        st.metric(
            label="📝 Total Words",
            value=f"{total_words:,}",
            delta="Content volume"
        )
    
    with col4:
        status = "🟢 Online" if data.get('status') == 'loaded' else "🔴 Offline"
        st.metric(
            label="🚀 System Status",
            value=status,
            delta="RAG System"
        )

def create_analytics_tab(data: Dict):
    """Create analytics and insights tab"""
    st.header("📊 System Analytics")
    
    chunks = data.get('chunks', [])
    if not chunks:
        st.warning("No data available for analytics.")
        return
    
    # Article distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Content Distribution")
        
        # Articles by chunk count
        article_chunks = {}
        for chunk in chunks:
            title = chunk.get('article_title', 'Unknown')
            article_chunks[title] = article_chunks.get(title, 0) + 1
        
        df_articles = pd.DataFrame([
            {'Article': title, 'Chunks': count}
            for title, count in sorted(article_chunks.items(), key=lambda x: x[1], reverse=True)[:10]
        ])
        
        fig = px.bar(df_articles, x='Chunks', y='Article', orientation='h',
                     title="Top 10 Articles by Chunk Count",
                     color='Chunks', color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📏 Content Length Analysis")
        
        # Content length distribution
        lengths = [len(chunk.get('content', '').split()) for chunk in chunks]
        
        fig = px.histogram(x=lengths, nbins=20,
                          title="Distribution of Chunk Lengths (words)",
                          labels={'x': 'Words per Chunk', 'y': 'Frequency'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_length = np.mean(lengths) if lengths else 0
        st.metric("Average Chunk Length", f"{avg_length:.0f} words")
    
    with col2:
        median_length = np.median(lengths) if lengths else 0
        st.metric("Median Chunk Length", f"{median_length:.0f} words")
    
    with col3:
        max_length = max(lengths) if lengths else 0
        st.metric("Longest Chunk", f"{max_length:.0f} words")

def create_advanced_evaluation_tab(data: Dict, rag_system):
    """Create advanced evaluation tab with comprehensive testing suite"""
    st.header("🧪 Advanced RAG Evaluation Suite")
    
    if not ADVANCED_EVALUATION_AVAILABLE:
        st.error("⚠️ Advanced evaluation features require additional dependencies. Please install: `pip install sentence-transformers scikit-learn`")
        return
    
    st.markdown("""
    This advanced evaluation suite provides comprehensive testing of the RAG system using cutting-edge evaluation techniques:
    
    - 🎯 **Adversarial Testing**: Challenge the system with ambiguous, negated, and multi-hop questions
    - 🔬 **Ablation Studies**: Compare different retrieval methods and parameter combinations  
    - ⚖️ **LLM-as-Judge**: Automated evaluation of factual accuracy, completeness, and coherence
    - 🎯 **Confidence Calibration**: Measure how well confidence scores correlate with correctness
    - 🆕 **Novel Metrics**: Custom metrics for entity coverage, diversity, and hallucination detection
    - 🔍 **Error Analysis**: Categorize and analyze failure modes with visualizations
    """)
    
    # Evaluation configuration
    st.subheader("🔧 Evaluation Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_questions = st.slider("Number of Test Questions", min_value=10, max_value=100, value=25)
    
    with col2:
        evaluation_type = st.selectbox("Evaluation Type", [
            "Quick Assessment",
            "Comprehensive Analysis", 
            "Adversarial Focus",
            "Parameter Optimization"
        ])
    
    with col3:
        auto_generate = st.checkbox("Auto-generate Questions", value=True, 
                                   help="Generate questions from the corpus automatically")
    
    # Sample questions for evaluation
    if auto_generate:
        # Generate sample questions from the corpus
        sample_questions = generate_sample_questions(data.get('chunks', []), num_questions)
    else:
        st.subheader("📝 Enter Custom Questions")
        custom_questions = st.text_area(
            "Enter questions (one per line):",
            placeholder="What is artificial intelligence?\nHow does machine learning work?\n...",
            height=150
        )
        sample_questions = [q.strip() for q in custom_questions.split('\n') if q.strip()]
    
    # Display sample questions
    if sample_questions:
        with st.expander(f"📋 Preview Questions ({len(sample_questions)} total)"):
            for i, q in enumerate(sample_questions[:10], 1):
                st.write(f"{i}. {q}")
            if len(sample_questions) > 10:
                st.write(f"... and {len(sample_questions) - 10} more questions")
    
    # Run evaluation
    if st.button("🚀 Run Advanced Evaluation", type="primary") and sample_questions:
        if not RAG_AVAILABLE or not rag_system:
            st.error("RAG system not available. Please check system initialization.")
            return
        
        # Initialize evaluation components
        evaluator = AdvancedEvaluationSuite()
        visualizer = EvaluationVisualizer()
        
        # Run comprehensive evaluation
        with st.spinner("🧪 Running advanced evaluation suite... This may take a few minutes."):
            comprehensive_results = evaluator.run_comprehensive_evaluation(
                sample_questions, 
                rag_system, 
                num_samples=min(num_questions, len(sample_questions))
            )
        
        # Store results in session state for persistence
        st.session_state.evaluation_results = comprehensive_results
        
        # Display results
        st.success("✅ Evaluation completed! Results are displayed below.")
    
    # Display cached results if available
    if 'evaluation_results' in st.session_state:
        st.markdown("---")
        visualizer = EvaluationVisualizer()
        visualizer.create_comprehensive_dashboard(st.session_state.evaluation_results)
        
        # Export options
        st.subheader("📊 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download JSON"):
                results_json = json.dumps(st.session_state.evaluation_results, indent=2, default=str)
                st.download_button(
                    label="💾 Download Evaluation Results",
                    data=results_json,
                    file_name=f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📈 Generate Report"):
                st.info("Comprehensive PDF report generation coming soon!")
        
        with col3:
            if st.button("🔄 Clear Results"):
                if 'evaluation_results' in st.session_state:
                    del st.session_state.evaluation_results
                st.rerun()


def generate_sample_questions(chunks: List[Dict], num_questions: int = 25) -> List[str]:
    """Generate sample questions from the corpus"""
    if not chunks:
        return [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the benefits of deep learning?",
            "What is natural language processing?",
            "How do neural networks function?"
        ]
    
    questions = []
    question_templates = [
        "What is {topic}?",
        "How does {topic} work?",
        "What are the benefits of {topic}?",
        "What are the applications of {topic}?",
        "How can {topic} be improved?"
    ]
    
    # Extract key topics from chunks
    topics = set()
    for chunk in chunks[:50]:  # Limit for performance
        content = chunk.get('content', '')
        title = chunk.get('article_title', '')
        
        # Simple topic extraction (in practice, use NLP)
        import re
        # Extract capitalized phrases as potential topics  
        found_topics = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content + ' ' + title)
        topics.update(found_topics[:3])  # Limit topics per chunk
    
    # Generate questions
    topic_list = list(topics)[:20]  # Limit total topics
    
    for i in range(min(num_questions, len(topic_list) * len(question_templates))):
        topic = topic_list[i % len(topic_list)]
        template = question_templates[i % len(question_templates)]
        question = template.format(topic=topic)
        if question not in questions:
            questions.append(question)
    
    # Add some challenging questions
    challenging_questions = [
        "What are the ethical implications of AI development?",
        "How do different machine learning algorithms compare?",
        "What are the limitations of current AI systems?",
        "How can bias in AI systems be mitigated?",
        "What is the future of artificial intelligence?"
    ]
    
    questions.extend(challenging_questions[:max(0, num_questions - len(questions))])
    
    return questions[:num_questions]


def main():
    """Main application"""
    # Load system data
    with st.spinner("🔄 Loading system data..."):
        data = load_system_data()
    
    # Initialize RAG system
    rag_system = None
    if RAG_AVAILABLE:
        with st.spinner("🚀 Initializing RAG system..."):
            rag_system = initialize_rag_system()
    
    # Create dashboard
    create_dashboard(data)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Search", 
        "📊 Analytics", 
        "🧪 Advanced Evaluation", 
        "⚙️ Settings", 
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("🔍 Intelligent Search")
        
        # Search interface
        col1, col2 = st.columns([0.8, 0.2])
        
        with col1:
            query = st.text_input(
                "Enter your search query:",
                placeholder="e.g., 'artificial intelligence machine learning'",
                help="Search across all articles and chunks in the knowledge base"
            )
        
        with col2:
            top_k = st.selectbox("Results", [5, 10, 15, 20], index=1)
        
        # Advanced search options
        with st.expander("🔧 Advanced Options"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_mode = st.selectbox(
                    "Search Mode",
                    ["Hybrid", "Keyword Only", "Semantic Only"],
                    help="Choose search strategy"
                )
            
            with col2:
                include_scores = st.checkbox(
                    "Show Score Details",
                    help="Display detailed scoring information"
                )
            
            with col3:
                filter_articles = st.multiselect(
                    "Filter by Articles",
                    options=list(set(chunk.get('article_title', '') 
                               for chunk in data.get('chunks', []))),
                    help="Limit search to specific articles"
                )
        
        # Perform search
        if query:
            with st.spinner("🔎 Searching knowledge base..."):
                start_time = time.time()
                
                chunks_to_search = data.get('chunks', [])
                if filter_articles:
                    chunks_to_search = [
                        chunk for chunk in chunks_to_search 
                        if chunk.get('article_title', '') in filter_articles
                    ]
                
                results = perform_hybrid_search(query, chunks_to_search, top_k)
                search_time = time.time() - start_time
            
            # Display results
            if results:
                st.success(f"✅ Search completed in {search_time:.2f} seconds")
                display_search_results(results, query)
            else:
                st.warning("🤔 No results found. Try different keywords.")
        
        elif data.get('status') == 'loaded':
            # Show sample queries
            st.info("💡 **Try these sample queries:**")
            sample_queries = [
                "artificial intelligence",
                "machine learning algorithms",
                "computer security threats",
                "software engineering practices",
                "deep learning neural networks"
            ]
            
            for query_example in sample_queries:
                if st.button(f"🔍 {query_example}", key=f"sample_{query_example}"):
                    st.rerun()
    
    with tab2:
        create_analytics_tab(data)
    
    with tab3:
        create_advanced_evaluation_tab(data, rag_system)
    
    with tab4:
        st.header("⚙️ System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Search Parameters")
            
            # Configuration options
            config = data.get('config', {})
            
            chunk_size = st.slider(
                "Chunk Size (words)",
                min_value=100, max_value=500,
                value=config.get('chunk_size', 300),
                help="Size of text chunks for processing"
            )
            
            overlap = st.slider(
                "Chunk Overlap",
                min_value=0, max_value=100,
                value=config.get('overlap', 50),
                help="Word overlap between chunks"
            )
            
        with col2:
            st.subheader("📊 System Information")
            
            # System stats
            stats = data.get('stats', {})
            
            if stats:
                st.json(stats)
            else:
                st.info("No system statistics available")
        
        # Data management
        st.subheader("💾 Data Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reload Data"):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared!")
        
        with col3:
            if st.button("📊 Export Results"):
                st.info("Export functionality coming soon!")
    
    with tab5:
        st.header("ℹ️ About Hybrid RAG System")
        
        st.markdown("""
        ### 🎯 System Overview
        
        This Hybrid RAG (Retrieval-Augmented Generation) system combines multiple search 
        strategies to provide comprehensive information retrieval:
        
        **🔍 Search Methods:**
        - **Keyword Search**: Traditional BM25-based sparse retrieval
        - **Semantic Search**: Dense vector similarity using embeddings
        - **Hybrid Fusion**: Combines both methods using Reciprocal Rank Fusion (RRF)
        
        **📚 Knowledge Base:**
        - Source: Wikipedia articles
        - Processing: Intelligent chunking with overlap
        - Coverage: Multiple domains including AI, technology, science
        
        **🚀 Features:**
        - Real-time search with sub-second response times
        - Score breakdown and ranking explanations
        - Interactive visualizations and analytics
        - Configurable search parameters
        
        **🛠️ Technical Stack:**
        - Backend: Python with FAISS for vector search
        - Frontend: Streamlit with custom CSS
        - Models: Sentence Transformers for embeddings
        - Storage: JSON-based with optional vector databases
        """)
        
        st.markdown("---")
        st.markdown("**Version:** 2.0.0 | **Last Updated:** February 2026")

if __name__ == "__main__":
    main()