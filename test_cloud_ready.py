#!/usr/bin/env python3
"""
Test script for cloud-ready Resume Relevance Check System
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    try:
        import streamlit as st
        from app import SimpleResumeAnalyzer
        from vector_search import VectorSearchEngine
        from database import ResumeDatabase
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_components():
    """Test individual components"""
    try:
        # Test analyzer
        from app import SimpleResumeAnalyzer
        analyzer = SimpleResumeAnalyzer()
        print("✅ Analyzer initialized")

        # Test vector search with cloud settings
        from vector_search import VectorSearchEngine
        vs = VectorSearchEngine(use_persistent=False)
        print("✅ Vector search (in-memory) initialized")

        # Test database with cloud settings
        from database import ResumeDatabase
        db = ResumeDatabase(use_memory=True)
        print("✅ Database (in-memory) initialized")

        return True
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        from app import SimpleResumeAnalyzer

        analyzer = SimpleResumeAnalyzer()

        # Test with sample data
        sample_resume = """
        John Doe
        Software Engineer

        Experience:
        - Python Developer at Tech Corp (2020-Present)
        - Java Developer at Startup Inc (2018-2020)

        Skills:
        - Python, Java, JavaScript
        - Machine Learning, Data Science
        - SQL, MongoDB
        - AWS, Docker
        """

        sample_jd = """
        Senior Python Developer

        Requirements:
        - 3+ years Python experience
        - Machine Learning experience
        - AWS cloud experience
        - SQL database knowledge
        """

        result = analyzer.analyze_resume(sample_resume, sample_jd)

        if 'relevance_score' in result and 'verdict' in result:
            print(f"✅ Analysis successful: Score {result['relevance_score']}, Verdict: {result['verdict']}")
            return True
        else:
            print("❌ Analysis failed - missing expected fields")
            return False

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Cloud-Ready Resume Relevance Check System")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Component Test", test_components),
        ("Basic Functionality Test", test_basic_functionality)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready for Streamlit Cloud deployment.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())