#!/usr/bin/env python3
"""
Resume Relevance Check System - Dependency Check
"""

def check_dependencies():
    """Check if all required dependencies are installed."""
    missing_deps = []

    try:
        import streamlit
        print("✅ Streamlit")
    except ImportError:
        missing_deps.append("streamlit")

    try:
        import google.generativeai
        print("✅ Google Generative AI")
    except ImportError:
        missing_deps.append("google-generativeai")

    try:
        import fitz
        print("✅ PyMuPDF")
    except ImportError:
        missing_deps.append("PyMuPDF")

    try:
        from docx import Document
        print("✅ python-docx")
    except ImportError:
        missing_deps.append("python-docx")

    try:
        import chromadb
        print("✅ ChromaDB")
    except ImportError:
        missing_deps.append("chromadb")

    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers")
    except ImportError:
        missing_deps.append("sentence-transformers")

    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please run: pip install -r requirements.txt")
        return False

    print("✅ All dependencies installed!")
    return True

if __name__ == "__main__":
    success = check_dependencies()
    exit(0 if success else 1)