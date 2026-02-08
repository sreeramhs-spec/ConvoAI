#!/usr/bin/env python3
"""
Simple BM25 search using collected Wikipedia data.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

try:
    from rank_bm25 import BM25Okapi
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Try to download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except:
            print("Warning: Could not download NLTK punkt tokenizer")
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords')
        except:
            print("Warning: Could not download NLTK stopwords")
    
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not available, using simple tokenization")
    NLTK_AVAILABLE = False

class SimpleBM25Search:
    """Simple BM25 search using collected data."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.chunks = []
        self.bm25 = None
        self.tokenized_corpus = []
        
        # Load stopwords
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = self._get_basic_stopwords()
        else:
            self.stop_words = self._get_basic_stopwords()
    
    def _get_basic_stopwords(self):
        """Basic stopwords if NLTK is not available."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that',
            'the', 'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'would',
            'there', 'we', 'when', 'where', 'who', 'how', 'all', 'any', 'can',
            'could', 'should', 'would', 'might', 'must'
        }
    
    def tokenize(self, text):
        """Tokenize text with optional NLTK."""
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower())
            except:
                tokens = text.lower().split()
        else:
            # Simple tokenization
            tokens = text.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split()
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        return tokens
    
    def load_data(self):
        """Load processed chunks."""
        chunks_file = self.data_dir / "processed" / "chunks.json"
        
        if not chunks_file.exists():
            print(f"❌ Chunks file not found: {chunks_file}")
            print("Run collect_data.py first to collect data")
            return False
        
        print(f"📄 Loading data from: {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"✅ Loaded {len(self.chunks)} chunks")
        return True
    
    def build_index(self):
        """Build BM25 index."""
        if not self.chunks:
            print("❌ No chunks loaded. Run load_data() first.")
            return False
        
        print("🔨 Building BM25 index...")
        start_time = time.time()
        
        # Tokenize all chunks
        self.tokenized_corpus = []
        for chunk in self.chunks:
            tokens = self.tokenize(chunk['content'])
            self.tokenized_corpus.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        build_time = time.time() - start_time
        print(f"✅ Index built in {build_time:.2f} seconds")
        print(f"📊 Average tokens per chunk: {sum(len(tokens) for tokens in self.tokenized_corpus) / len(self.tokenized_corpus):.1f}")
        
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search using BM25."""
        if not self.bm25:
            print("❌ Index not built. Run build_index() first.")
            return []
        
        print(f"\n🔍 Searching: '{query}'")
        start_time = time.time()
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        print(f"Query tokens: {query_tokens}")
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top results
        top_indices = scores.argsort()[-top_k:][::-1]  # Top k in descending order
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                results.append((self.chunks[idx], float(scores[idx])))
        
        search_time = time.time() - start_time
        print(f"⏱️  Search completed in {search_time:.3f} seconds")
        print(f"📊 Found {len(results)} relevant results")
        
        return results
    
    def display_results(self, results: List[Tuple[Dict, float]]):
        """Display search results."""
        if not results:
            print("❌ No results found")
            return
        
        print("\n📋 SEARCH RESULTS:")
        print("=" * 80)
        
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n{i}. {chunk['article_title']} (Score: {score:.3f})")
            print(f"   Words: {chunk['word_count']} | Chunk ID: {chunk['chunk_id']}")
            print(f"   URL: {chunk['article_url']}")
            
            # Show first 200 characters of content
            content_preview = chunk['content'][:200].replace('\n', ' ')
            print(f"   Content: {content_preview}...")
            print("-" * 80)
    
    def interactive_search(self):
        """Interactive search interface."""
        print("\n🔍 INTERACTIVE BM25 SEARCH")
        print("=" * 50)
        print("Enter queries to search (type 'quit' to exit)")
        
        while True:
            try:
                query = input("\n🔍 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                results = self.search(query, top_k=5)
                self.display_results(results)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def run_sample_queries(searcher):
    """Run some sample queries."""
    print("\n🧪 RUNNING SAMPLE QUERIES")
    print("=" * 50)
    
    sample_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?",
        "Explain quantum computing",
        "What is cloud computing?",
        "Python programming language features",
        "Database management systems",
        "Computer security methods"
    ]
    
    for query in sample_queries[:5]:  # Run first 5 queries
        results = searcher.search(query, top_k=3)
        searcher.display_results(results)
        print("\n" + "="*80)
        time.sleep(1)  # Brief pause between queries

def main():
    """Main execution function."""
    print("🔍 BM25 SEARCH SYSTEM")
    print("=" * 50)
    
    # Initialize searcher
    searcher = SimpleBM25Search()
    
    # Load data
    if not searcher.load_data():
        return
    
    # Build index
    if not searcher.build_index():
        return
    
    print("\n📋 SYSTEM READY")
    print("=" * 50)
    print(f"• Chunks indexed: {len(searcher.chunks)}")
    print(f"• Articles covered: {len(set(chunk['article_title'] for chunk in searcher.chunks))}")
    print(f"• Total words: {sum(chunk['word_count'] for chunk in searcher.chunks):,}")
    
    # Option menu
    while True:
        print("\n🎯 SELECT AN OPTION:")
        print("1. Run sample queries")
        print("2. Interactive search")
        print("3. Exit")
        
        try:
            choice = input("\nChoice (1-3): ").strip()
            
            if choice == '1':
                run_sample_queries(searcher)
            elif choice == '2':
                searcher.interactive_search()
            elif choice == '3':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()