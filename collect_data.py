#!/usr/bin/env python3
"""
Simplified data collector that works without transformer dependencies.
"""

import json
import os
import requests
import time
import random
from pathlib import Path
from typing import List, Dict
import wikipedia
from bs4 import BeautifulSoup

class SimpleDataCollector:
    """Simplified data collection without transformer dependencies."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_sample_data(self, num_articles=20):
        """Collect a small sample of Wikipedia articles."""
        print(f"🔍 Collecting {num_articles} Wikipedia articles...")
        
        # Popular topics that should work reliably
        sample_topics = [
            "Artificial Intelligence", "Machine Learning", "Deep Learning",
            "Natural Language Processing", "Computer Science", "Python (programming language)",
            "Data Science", "Neural Network", "Algorithm", "Software Engineering",
            "Database", "Web Development", "Cloud Computing", "Cybersecurity",
            "Blockchain", "Internet of Things", "Virtual Reality", "Augmented Reality",
            "Big Data", "Quantum Computing", "Robotics", "Operating System",
            "Computer Graphics", "Human-Computer Interaction", "Information Theory"
        ]
        
        articles = []
        successful = 0
        
        # Shuffle and take subset
        random.shuffle(sample_topics)
        topics_to_try = sample_topics[:num_articles * 2]  # Try more than needed
        
        for i, topic in enumerate(topics_to_try):
            if successful >= num_articles:
                break
                
            try:
                print(f"  Collecting {i+1}/{len(topics_to_try)}: {topic}")
                
                # Get Wikipedia page
                page = wikipedia.page(topic)
                content = page.content
                
                # Skip if too short
                if len(content.split()) < 100:
                    print(f"    Skipping '{topic}' - too short")
                    continue
                
                article_data = {
                    'title': page.title,
                    'url': page.url,
                    'content': content,
                    'summary': page.summary,
                    'word_count': len(content.split()),
                    'collected_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                articles.append(article_data)
                successful += 1
                print(f"    ✅ Collected '{page.title}' ({article_data['word_count']} words)")
                
                # Be nice to Wikipedia
                time.sleep(0.5)
                
            except wikipedia.exceptions.DisambiguationError as e:
                # Try the first option
                try:
                    page = wikipedia.page(e.options[0])
                    content = page.content
                    
                    if len(content.split()) >= 100:
                        article_data = {
                            'title': page.title,
                            'url': page.url,
                            'content': content,
                            'summary': page.summary,
                            'word_count': len(content.split()),
                            'collected_at': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        articles.append(article_data)
                        successful += 1
                        print(f"    ✅ Collected '{page.title}' (disambiguation resolved)")
                        time.sleep(0.5)
                    
                except Exception:
                    print(f"    ❌ Failed to resolve disambiguation for '{topic}'")
                    
            except Exception as e:
                print(f"    ❌ Failed to collect '{topic}': {e}")
                continue
        
        # Save collected articles
        output_file = self.raw_dir / "wikipedia_articles.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Collected {successful} articles")
        print(f"💾 Saved to: {output_file}")
        
        return articles
    
    def process_articles(self, articles):
        """Process articles into chunks."""
        print("\n🔄 Processing articles into chunks...")
        
        all_chunks = []
        chunk_id = 0
        
        for article in articles:
            content = article['content']
            title = article['title']
            
            # Simple text chunking - split by paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                # Skip very short paragraphs
                if len(paragraph.split()) < 20:
                    continue
                
                chunk = {
                    'chunk_id': f"chunk_{chunk_id}",
                    'article_title': title,
                    'article_url': article['url'],
                    'paragraph_index': i,
                    'content': paragraph,
                    'word_count': len(paragraph.split()),
                    'char_count': len(paragraph)
                }
                
                all_chunks.append(chunk)
                chunk_id += 1
        
        # Save processed chunks
        output_file = self.processed_dir / "chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created {len(all_chunks)} chunks")
        print(f"💾 Saved to: {output_file}")
        
        # Save metadata
        metadata = {
            'num_articles': len(articles),
            'num_chunks': len(all_chunks),
            'total_words': sum(chunk['word_count'] for chunk in all_chunks),
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = self.processed_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return all_chunks
    
    def create_sample_queries(self, articles):
        """Create sample queries based on collected articles."""
        print("\n❓ Creating sample queries...")
        
        queries = []
        
        # Extract key topics from article titles
        for article in articles[:10]:  # Use first 10 articles
            title = article['title']
            summary = article.get('summary', '')[:200]  # First 200 chars of summary
            
            # Create different types of queries
            queries.extend([
                f"What is {title.lower()}?",
                f"Explain {title.lower()}",
                f"How does {title.lower().split('(')[0].strip()} work?",
                f"Applications of {title.lower().split('(')[0].strip()}",
            ])
        
        # Add some generic queries
        queries.extend([
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are neural networks?",
            "Explain deep learning concepts",
            "What is natural language processing?",
            "How do algorithms work?",
            "What is computer science?",
            "Explain data structures",
            "What is software engineering?",
            "How does cloud computing work?"
        ])
        
        # Remove duplicates and shuffle
        queries = list(set(queries))
        random.shuffle(queries)
        
        # Save queries
        queries_file = self.processed_dir / "sample_queries.json"
        with open(queries_file, 'w', encoding='utf-8') as f:
            json.dump(queries[:50], f, indent=2)  # Save first 50 unique queries
        
        print(f"✅ Created {len(queries[:50])} sample queries")
        print(f"💾 Saved to: {queries_file}")
        
        return queries[:50]

def main():
    """Main execution function."""
    print("📊 SIMPLE DATA COLLECTION FOR RAG SYSTEM")
    print("="*60)
    
    collector = SimpleDataCollector()
    
    # Collect articles
    articles = collector.collect_sample_data(num_articles=25)
    
    # Process into chunks
    chunks = collector.process_articles(articles)
    
    # Create sample queries
    queries = collector.create_sample_queries(articles)
    
    print("\n📋 COLLECTION SUMMARY")
    print("="*60)
    print(f"• Articles collected: {len(articles)}")
    print(f"• Chunks created: {len(chunks)}")
    print(f"• Sample queries: {len(queries)}")
    print(f"• Data directory: {collector.data_dir.absolute()}")
    
    print("\n📁 FILES CREATED:")
    for file_path in collector.raw_dir.glob("*.json"):
        print(f"  📄 {file_path}")
    for file_path in collector.processed_dir.glob("*.json"):
        print(f"  📄 {file_path}")
    
    print("\n✅ Data collection completed successfully!")
    print("\n🔧 NEXT STEPS:")
    print("1. Review collected data in the data/ directory")
    print("2. Run the demo system with: python demo.py")
    print("3. Try the BM25 search on collected data")

if __name__ == "__main__":
    main()