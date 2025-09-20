import google.generativeai as genai
import fitz  # PyMuPDF
from docx import Document
import json
import re
import os
from typing import Dict, Any, Optional, List
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced modules
try:
    from text_processor import EnhancedTextProcessor
    from database import ResumeDatabase
    from vector_search import VectorSearchEngine
    from langchain_analyzer import LangChainResumeAnalyzer
    ENHANCED_FEATURES_AVAILABLE = True
    logger.info("Enhanced modules imported successfully")
except ImportError as e:
    logger.warning(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False
    # Define dummy classes to prevent errors
    class EnhancedTextProcessor:
        def __init__(self): pass
    class ResumeDatabase:
        def __init__(self, *args, **kwargs): pass
    class VectorSearchEngine:
        def __init__(self, *args, **kwargs): pass
    class LangChainResumeAnalyzer:
        def __init__(self, *args, **kwargs): pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeAnalyzer:
    def __init__(self, api_key: str, use_enhanced_features: bool = True):
        """Initialize the Resume Analyzer with Gemini Pro API."""
        self.api_key = api_key
        genai.configure(api_key=api_key)

        # Try different models in order of preference
        self.models_to_try = [
            'gemini-1.5-flash',  # Cheaper and faster
            'gemini-1.5-pro',    # More capable but more expensive
            'gemini-2.0-flash',  # Latest fast model
        ]
        self.model = None
        self._initialize_model()

        # Initialize enhanced features if available
        self.use_enhanced_features = use_enhanced_features and ENHANCED_FEATURES_AVAILABLE
        self.text_processor = None
        self.database = None
        self.vector_search = None
        self.langchain_analyzer = None

        if self.use_enhanced_features:
            self._initialize_enhanced_features()

    def _initialize_enhanced_features(self):
        """Initialize enhanced features if available."""
        try:
            # Initialize enhanced text processor
            self.text_processor = EnhancedTextProcessor()
            logger.info("Enhanced text processor initialized")

            # Initialize database
            self.database = ResumeDatabase()
            logger.info("Database initialized")

            # Initialize vector search
            self.vector_search = VectorSearchEngine()
            logger.info("Vector search engine initialized")

            # Initialize LangChain analyzer
            self.langchain_analyzer = LangChainResumeAnalyzer(self.api_key)
            logger.info("LangChain analyzer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize enhanced features: {e}")
            self.use_enhanced_features = False
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            # Read PDF from uploaded file
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text()
            
            pdf_document.close()
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from uploaded DOCX file."""
        try:
            doc = Document(docx_file)
            text = []
            
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            
            return '\n'.join(text).strip()
        except Exception as e:
            st.error(f"Error extracting DOCX text: {str(e)}")
            return ""
    
    def extract_text_from_file(self, uploaded_file) -> str:
        """Extract text from uploaded file based on type."""
        if uploaded_file.type == "application/pdf":
            return self.extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self.extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload PDF or DOCX files.")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove extra whitespaces and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-\.\@\+\(\)]', ' ', text)
        return text.strip()
    
    def create_analysis_prompt(self, resume_text: str, jd_text: str) -> str:
        """Create the prompt for Gemini Pro analysis."""
        prompt = f"""
Analyze the following resume against the job description and provide a detailed evaluation.

RESUME CONTENT:
{resume_text[:4000]}  # Limit text to avoid token limits

JOB DESCRIPTION:
{jd_text[:2000]}

Please analyze and return ONLY a valid JSON response with this exact structure:
{{
    "relevance_score": <number between 0-100>,
    "verdict": "<High/Medium/Low>",
    "missing_skills": ["skill1", "skill2", "skill3"],
    "matching_skills": ["skill1", "skill2", "skill3"],
    "missing_certifications": ["cert1", "cert2"],
    "experience_match": "<brief description of experience alignment>",
    "education_match": "<brief description of education alignment>",
    "improvement_suggestions": ["suggestion1", "suggestion2", "suggestion3"],
    "key_strengths": ["strength1", "strength2", "strength3"],
    "overall_feedback": "<2-3 sentence overall assessment>"
}}

Scoring Guidelines:
- 90-100: Excellent match, ready for interview
- 70-89: Good match, minor gaps
- 50-69: Moderate match, some important gaps
- 30-49: Poor match, major gaps
- 0-29: Very poor match, not suitable

Focus on:
1. Technical skills match
2. Experience level alignment
3. Education requirements
4. Industry experience
5. Certification requirements

Return only the JSON, no additional text.
"""
        return prompt
    
    def _initialize_model(self):
        """Initialize the model, trying different models if one fails."""
        for model_name in self.models_to_try:
            try:
                self.model = genai.GenerativeModel(model_name)
                print(f"Successfully initialized model: {model_name}")
                return
            except Exception as e:
                print(f"Failed to initialize {model_name}: {e}")
                continue
        
        # If all models fail, raise an error
        raise Exception("Unable to initialize any Gemini model. Please check your API key and billing status.")

    def analyze_resume(self, resume_text: str, jd_text: str, use_enhanced: bool = None) -> Dict[str, Any]:
        """Analyze resume against job description using Gemini Pro."""
        # Determine whether to use enhanced features
        use_enhanced = use_enhanced if use_enhanced is not None else self.use_enhanced_features

        # Use enhanced analysis if available and requested
        if use_enhanced and self.langchain_analyzer:
            logger.info("Using enhanced LangChain analysis")
            return self._analyze_with_langchain(resume_text, jd_text)

        # Fall back to basic analysis
        return self._analyze_basic(resume_text, jd_text)

    def _analyze_basic(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Basic analysis using direct Gemini API calls."""
        try:
            # Clean the texts
            clean_resume = self.clean_text(resume_text)
            clean_jd = self.clean_text(jd_text)

            # Create prompt
            prompt = self.create_analysis_prompt(clean_resume, clean_jd)

            # Generate response with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    break
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        if attempt < max_retries - 1:
                            wait_time = min(30, 2 ** attempt)  # Exponential backoff
                            print(f"Quota exceeded, retrying in {wait_time} seconds...")
                            import time
                            time.sleep(wait_time)
                            continue
                        else:
                            # If quota exceeded after retries, show error
                            print("Quota exceeded, please check your API limits")
                            st.error("❌ API quota exceeded. Please check your billing or try again later.")
                            return self._get_fallback_response()
                    else:
                        raise e

            # Parse JSON response
            try:
                # Extract JSON from response
                response_text = response.text.strip()

                # Remove any markdown formatting if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]

                result = json.loads(response_text)

                # Validate required fields
                required_fields = [
                    'relevance_score', 'verdict', 'missing_skills',
                    'matching_skills', 'improvement_suggestions', 'overall_feedback'
                ]

                for field in required_fields:
                    if field not in result:
                        result[field] = self._get_default_value(field)

                # Ensure score is within range
                if result['relevance_score'] > 100:
                    result['relevance_score'] = 100
                elif result['relevance_score'] < 0:
                    result['relevance_score'] = 0

                return result

            except json.JSONDecodeError as e:
                st.error(f"Error parsing AI response: {str(e)}")
                return self._get_fallback_response()

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            return self._get_fallback_response()

    def _analyze_with_langchain(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Enhanced analysis using LangChain."""
        try:
            # Use LangChain analyzer
            result = self.langchain_analyzer.analyze_resume(resume_text, jd_text)

            # Add enhanced features if available
            if self.text_processor:
                # Extract additional entities and skills
                entities = self.text_processor.extract_entities(resume_text)
                skills = self.text_processor.extract_skills(resume_text)

                # Enhance the result with additional data
                if 'matching_skills' in result:
                    # Merge with extracted skills
                    existing_skills = set(result['matching_skills'])
                    extracted_skills = set(skills)
                    result['matching_skills'] = list(existing_skills.union(extracted_skills))

                result['extracted_entities'] = entities
                result['enhanced_skills'] = skills

            # Store in database if available
            if self.database:
                try:
                    self.database.save_analysis_result(result)
                except Exception as e:
                    logger.warning(f"Failed to save to database: {e}")

            # Add to vector search if available
            if self.vector_search:
                try:
                    self.vector_search.add_resume(resume_text, {'filename': 'analysis_input'})
                except Exception as e:
                    logger.warning(f"Failed to add to vector search: {e}")

            return result

        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            # Fall back to basic analysis
            return self._analyze_basic(resume_text, jd_text)
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields."""
        defaults = {
            'relevance_score': 0,
            'verdict': 'Low',
            'missing_skills': ['Unable to analyze'],
            'matching_skills': ['Unable to analyze'],
            'missing_certifications': [],
            'experience_match': 'Unable to analyze',
            'education_match': 'Unable to analyze',
            'improvement_suggestions': ['Please try again'],
            'key_strengths': ['Unable to analyze'],
            'overall_feedback': 'Analysis failed. Please try again.'
        }
        return defaults.get(field, 'N/A')
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Return fallback response when analysis fails."""
        return {
            'relevance_score': 0,
            'verdict': 'Low',
            'missing_skills': ['Analysis failed'],
            'matching_skills': ['Analysis failed'],
            'missing_certifications': [],
            'experience_match': 'Analysis failed',
            'education_match': 'Analysis failed',
            'improvement_suggestions': ['Please try uploading again'],
            'key_strengths': ['Analysis failed'],
            'overall_feedback': 'Unable to analyze resume. Please check file format and try again.'
        }
    
    def batch_analyze(self, resume_files, jd_text: str, use_enhanced: bool = None) -> list:
        """Analyze multiple resumes against a single job description."""
        results = []
        use_enhanced = use_enhanced if use_enhanced is not None else self.use_enhanced_features

        progress_bar = st.progress(0)

        for i, resume_file in enumerate(resume_files):
            st.write(f"Processing: {resume_file.name}")

            # Extract resume text
            resume_text = self.extract_text_from_file(resume_file)

            if resume_text:
                # Analyze resume
                analysis = self.analyze_resume(resume_text, jd_text, use_enhanced)
                analysis['filename'] = resume_file.name
                results.append(analysis)

                # Save to database if enhanced features are enabled
                if use_enhanced and self.database:
                    try:
                        self.database.save_analysis_result(analysis)
                    except Exception as e:
                        logger.warning(f"Failed to save batch result to database: {e}")
            else:
                # Add failed analysis
                failed_analysis = self._get_fallback_response()
                failed_analysis['filename'] = resume_file.name
                results.append(failed_analysis)

            # Update progress
            progress_bar.progress((i + 1) / len(resume_files))

        # Save batch summary if enhanced features are enabled
        if use_enhanced and self.database and results:
            try:
                self.database.save_batch_analysis("Batch Analysis", results)
            except Exception as e:
                logger.warning(f"Failed to save batch summary: {e}")

        return results

    def get_enhanced_features_status(self) -> Dict[str, bool]:
        """Get status of enhanced features."""
        return {
            'enhanced_features_available': ENHANCED_FEATURES_AVAILABLE,
            'text_processor': self.text_processor is not None,
            'database': self.database is not None,
            'vector_search': self.vector_search is not None,
            'langchain_analyzer': self.langchain_analyzer is not None
        }

    def search_similar_resumes(self, job_description: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for resumes similar to a job description using vector search."""
        if not self.vector_search:
            logger.warning("Vector search not available")
            return []

        try:
            return self.vector_search.find_similar_resumes(job_description, limit)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.database:
            return {'error': 'Database not available'}

        try:
            return self.database.get_statistics()
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}

    def export_database_to_csv(self) -> str:
        """Export database contents to CSV."""
        if not self.database:
            return "Database not available"

        try:
            return self.database.export_to_csv()
        except Exception as e:
            logger.error(f"Failed to export database: {e}")
            return f"Export failed: {e}"
    
    def get_verdict_color(self, verdict: str) -> str:
        """Get color for verdict display."""
        colors = {
            'High': '#28a745',    # Green
            'Medium': '#ffc107',  # Yellow
            'Low': '#dc3545'      # Red
        }
        return colors.get(verdict, '#6c757d')
    
    def get_score_color(self, score: int) -> str:
        """Get color based on score range."""
        if score >= 80:
            return '#28a745'  # Green
        elif score >= 60:
            return '#ffc107'  # Yellow
        elif score >= 40:
            return '#fd7e14'  # Orange
        else:
            return '#dc3545'  # Red


# Helper functions for Streamlit app
def format_skills_display(skills: list, skill_type: str = "matching") -> str:
    """Format skills for display in Streamlit."""
    if not skills:
        return "None identified"
    
    if skill_type == "matching":
        color = "#28a745"  # Green
        icon = "✅"
    else:  # missing
        color = "#dc3545"  # Red
        icon = "❌"
    
    formatted_skills = []
    for skill in skills[:10]:  # Limit to 10 skills for display
        formatted_skills.append(f"<span style='color: {color}'>{icon} {skill}</span>")
    
    return "<br>".join(formatted_skills)


def export_results_to_csv(results: list) -> str:
    """Convert results to CSV format for download."""
    import pandas as pd
    
    # Flatten results for CSV
    flattened_results = []
    for result in results:
        flat_result = {
            'filename': result.get('filename', 'Unknown'),
            'relevance_score': result.get('relevance_score', 0),
            'verdict': result.get('verdict', 'Low'),
            'matching_skills': ', '.join(result.get('matching_skills', [])),
            'missing_skills': ', '.join(result.get('missing_skills', [])),
            'experience_match': result.get('experience_match', 'N/A'),
            'education_match': result.get('education_match', 'N/A'),
            'overall_feedback': result.get('overall_feedback', 'N/A')
        }
        flattened_results.append(flat_result)
    
    df = pd.DataFrame(flattened_results)
    return df.to_csv(index=False)