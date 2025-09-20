#!/usr/bin/env python3
"""
Health Check Script for Resume Relevance Check System
"""

import requests
import time
import sys
import os

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")

    dependencies = [
        ('streamlit', 'Streamlit'),
        ('google.generativeai', 'Google Generative AI'),
        ('fitz', 'PyMuPDF'),
        ('docx', 'python-docx'),
        ('chromadb', 'ChromaDB'),
        ('sentence_transformers', 'Sentence Transformers'),
    ]

    missing_deps = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - MISSING")
            missing_deps.append(name)

    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("✅ All dependencies installed!")
    return True

def check_api_key():
    """Check if Gemini API key is configured."""
    print("\n🔑 Checking API configuration...")

    # Load .env file explicitly
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("  ⚠️  python-dotenv not available, checking environment directly")

    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("  ❌ GEMINI_API_KEY not found in environment")
        print("  Please create .env file with: GEMINI_API_KEY=your_api_key_here")
        return False

    if api_key == 'demo_key' or len(api_key) < 20:
        print("  ❌ GEMINI_API_KEY appears to be invalid or demo key")
        print("  Please set your actual Gemini API key from: https://makersuite.google.com/app/apikey")
        return False

    # Test the API key by trying to initialize Gemini
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # Try to create a model to verify the key works
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("  ✅ Gemini API key configured and working")
        return True
    except Exception as e:
        print(f"  ❌ Gemini API key test failed: {str(e)}")
        print("  Please verify your API key is correct")
        return False

def check_database():
    """Check if database is accessible."""
    print("\n💾 Checking database...")

    try:
        import sqlite3
        conn = sqlite3.connect('resume_analysis.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()

        print(f"  ✅ Database accessible ({len(tables)} tables)")
        return True
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        return False

def check_streamlit_server(port=8501):
    """Check if Streamlit server is running."""
    print(f"\n🌐 Checking Streamlit server on port {port}...")

    try:
        response = requests.get(f'http://localhost:{port}', timeout=5)
        if response.status_code == 200:
            print(f"  ✅ Streamlit server running on port {port}")
            return True
        else:
            print(f"  ❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Cannot connect to server: {e}")
        print(f"  💡 Make sure the server is running: streamlit run app.py --server.port={port}")
        return False

def test_basic_functionality():
    """Test basic application functionality."""
    print("\n🧪 Testing basic functionality...")

    try:
        from app import SimpleResumeAnalyzer

        # Test analyzer initialization
        analyzer = SimpleResumeAnalyzer()
        print("  ✅ Analyzer initialized successfully")

        # Test basic text extraction
        test_text = "This is a test resume with Python and AI skills."
        result = analyzer.analyze_resume(test_text, "Looking for Python developer")
        print("  ✅ Basic analysis working")

        return True
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def main():
    """Run all health checks."""
    print("🏥 Resume Relevance Check System - Health Check")
    print("=" * 50)

    checks = [
        ("Dependencies", check_dependencies),
        ("API Key", check_api_key),
        ("Database", check_database),
        ("Basic Functionality", test_basic_functionality),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results.append((name, False))

    # Check Streamlit server separately
    port = int(os.getenv('STREAMLIT_SERVER_PORT', 8501))
    server_result = check_streamlit_server(port)
    results.append(("Streamlit Server", server_result))

    # Summary
    print("\n" + "=" * 50)
    print("📊 HEALTH CHECK SUMMARY")
    print("=" * 50)

    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
        if not result:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - System is healthy!")
        print("🚀 Ready for production deployment")
    else:
        print("⚠️  SOME CHECKS FAILED - Please fix issues before deployment")
        print("📖 Check the deployment guide for troubleshooting")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())