#!/usr/bin/env python3
"""
Test script for enhanced Resume Relevance Check System
"""

from resume_analyzer import ResumeAnalyzer
import os

def test_enhanced_features():
    """Test all enhanced features."""
    print("🚀 Testing Enhanced Resume Relevance Check System")
    print("=" * 60)

    # Initialize analyzer with enhanced features
    api_key = os.getenv('GEMINI_API_KEY', 'test_key')
    analyzer = ResumeAnalyzer(api_key=api_key, use_enhanced_features=True)

    # Check enhanced features status
    print("\n📊 Enhanced Features Status:")
    status = analyzer.get_enhanced_features_status()
    for feature, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {feature}: {status_icon}")

    # Test basic analysis
    print("\n🔍 Testing Basic Analysis:")
    test_resume = """
    I am a Python developer with 5 years of experience in machine learning,
    data science, and web development. I have worked with TensorFlow, PyTorch,
    and scikit-learn. I hold a Master's degree in Computer Science.
    """

    test_jd = """
    We are looking for a Python developer with ML experience.
    Required skills: Python, TensorFlow, machine learning.
    Preferred: Master's degree in Computer Science.
    """

    try:
        result = analyzer.analyze_resume(test_resume, test_jd, use_enhanced=False)
        score = result.get('relevance_score', 0)
        verdict = result.get('verdict', 'Unknown')
        print(f"  ✅ Basic analysis successful: Score {score}/100 ({verdict})")
    except Exception as e:
        print(f"  ❌ Basic analysis failed: {e}")

    # Test enhanced analysis if available
    if status.get('langchain_analyzer', False):
        print("\n🤖 Testing Enhanced Analysis:")
        try:
            result = analyzer.analyze_resume(test_resume, test_jd, use_enhanced=True)
            score = result.get('relevance_score', 0)
            verdict = result.get('verdict', 'Unknown')
            print(f"  ✅ Enhanced analysis successful: Score {score}/100 ({verdict})")

            # Check for enhanced features in result
            if 'extracted_entities' in result:
                print("  ✅ Entity extraction working")
            if 'enhanced_skills' in result:
                print("  ✅ Enhanced skills extraction working")

        except Exception as e:
            print(f"  ❌ Enhanced analysis failed: {e}")

    # Test vector search if available
    if status.get('vector_search', False):
        print("\n🔎 Testing Vector Search:")
        try:
            # Add test data to vector search
            analyzer.vector_search.add_resume(test_resume, {'filename': 'test_resume'})
            analyzer.vector_search.add_job_description(test_jd, {'title': 'Test JD'})

            # Search for similar resumes
            similar = analyzer.search_similar_resumes(test_jd, limit=3)
            print(f"  ✅ Vector search working: Found {len(similar)} similar resumes")

        except Exception as e:
            print(f"  ❌ Vector search failed: {e}")

    # Test database if available
    if status.get('database', False):
        print("\n💾 Testing Database:")
        try:
            # Save test result
            test_result = {
                'filename': 'test_resume.pdf',
                'relevance_score': 85,
                'verdict': 'High',
                'matching_skills': ['Python', 'Machine Learning'],
                'missing_skills': ['AWS'],
                'experience_match': 'Good match',
                'education_match': 'Good match',
                'improvement_suggestions': ['Add cloud certifications'],
                'overall_feedback': 'Strong technical background'
            }

            result_id = analyzer.database.save_analysis_result(test_result)
            print(f"  ✅ Database save working: Result ID {result_id}")

            # Get stats
            stats = analyzer.database.get_statistics()
            print(f"  ✅ Database stats working: {stats.get('total_analyses', 0)} total analyses")

        except Exception as e:
            print(f"  ❌ Database failed: {e}")

    print("\n" + "=" * 60)
    print("🎉 Enhanced System Test Complete!")
    print("\n💡 Your system now includes:")
    print("  • spaCy for advanced text processing")
    print("  • NLTK for linguistic analysis")
    print("  • LangChain for structured AI workflows")
    print("  • ChromaDB for vector search")
    print("  • SQLite for persistent storage")
    print("  • Enhanced entity extraction and skill recognition")

if __name__ == "__main__":
    test_enhanced_features()