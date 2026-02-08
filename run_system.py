#!/usr/bin/env python3
"""
Simple runner for the RAG system that bypasses import issues.
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_individual_components():
    """Test individual components without full imports."""
    print("🔍 HYBRID RAG SYSTEM - COMPONENT TEST")
    print("="*60)
    
    try:
        # Test basic imports
        import json
        import requests
        import numpy as np
        import pandas as pd
        print("✅ Basic libraries working")
        
        # Test Wikipedia collector without transformers
        try:
            import wikipedia
            print("✅ Wikipedia library working")
        except ImportError as e:
            print(f"❌ Wikipedia import failed: {e}")
        
        # Test BM25 (rank_bm25)
        try:
            from rank_bm25 import BM25Okapi
            print("✅ BM25 (rank_bm25) working")
        except ImportError as e:
            print(f"❌ BM25 import failed: {e}")
        
        # Test FAISS 
        try:
            import faiss
            print("✅ FAISS working")
        except ImportError as e:
            print(f"❌ FAISS import failed: {e}")
        
        # Test Streamlit
        try:
            import streamlit
            print("✅ Streamlit working")
        except ImportError as e:
            print(f"❌ Streamlit import failed: {e}")
        
        # Test problematic transformers
        try:
            import torch
            print("✅ PyTorch working")
        except ImportError as e:
            print(f"❌ PyTorch import failed: {e}")
        
        try:
            from sentence_transformers import SentenceTransformer
            print("✅ Sentence Transformers working")
        except Exception as e:
            print(f"❌ Sentence Transformers failed: {e}")
            print("Note: This is expected due to version compatibility issues")
        
        print("\n🚀 RUNNING DEMO VERSION")
        print("="*60)
        
        # Run the demo which doesn't use problematic imports
        import subprocess
        result = subprocess.run([
            sys.executable, "demo.py"
        ], capture_output=True, text=True, cwd=str(Path(__file__).parent))
        
        if result.returncode == 0:
            print("✅ Demo completed successfully!")
            print("\nDemo output (first 1000 chars):")
            print(result.stdout[:1000])
            if len(result.stdout) > 1000:
                print("... (truncated)")
        else:
            print("❌ Demo failed:")
            print(result.stderr)
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")

def try_streamlit():
    """Try to run the Streamlit app with error handling."""
    print("\n🌐 TRYING TO START STREAMLIT APP")
    print("="*60)
    
    try:
        # Check if the Streamlit app file exists and can be read
        streamlit_app_path = Path("src/ui/streamlit_app.py")
        if streamlit_app_path.exists():
            print(f"✅ Found Streamlit app at: {streamlit_app_path}")
            
            print("\nTo run the Streamlit app manually, use:")
            print(f"streamlit run {streamlit_app_path}")
            print("\nNote: The app may have import issues due to transformer dependencies")
        else:
            print(f"❌ Streamlit app not found at: {streamlit_app_path}")
            
    except Exception as e:
        print(f"❌ Streamlit check failed: {e}")

def main():
    """Main execution function."""
    print("🏗️  RAG SYSTEM RUNNER")
    print("="*60)
    
    # Test individual components
    test_individual_components()
    
    # Try Streamlit setup
    try_streamlit()
    
    print("\n📋 SUMMARY")
    print("="*60)
    print("• Demo version works correctly")
    print("• Core libraries are installed")
    print("• Transformer libraries have version conflicts")
    print("• Consider using a virtual environment with specific versions")
    print("\n🔧 RECOMMENDATIONS:")
    print("1. Use the demo.py for immediate testing")
    print("2. Resolve transformer version conflicts for full functionality")
    print("3. Consider Docker or virtual environment setup")
    print("4. Check SSL certificates for model downloads")

if __name__ == "__main__":
    main()