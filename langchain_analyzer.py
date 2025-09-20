import json
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import BaseOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False
    # Define dummy classes
    class ChatGoogleGenerativeAI:
        pass
    class PromptTemplate:
        pass
    class LLMChain:
        pass
    class BaseOutputParser:
        pass

class ResumeAnalysisOutputParser(BaseOutputParser):
    """Custom output parser for resume analysis results."""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the LLM output into structured format."""
        try:
            # Try to extract JSON from the response
            text = text.strip()

            # Remove markdown formatting if present
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]

            # Parse JSON
            result = json.loads(text.strip())

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
            logger.error(f"JSON parsing error: {e}")
            return self._get_fallback_response()
        except Exception as e:
            logger.error(f"Output parsing error: {e}")
            return self._get_fallback_response()

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
        """Return fallback response when parsing fails."""
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

    def get_format_instructions(self) -> str:
        """Return format instructions for the LLM."""
        return """Return your response as a valid JSON object with the following structure:
{
  "relevance_score": <number between 0-100>,
  "verdict": "<High|Medium|Low>",
  "matching_skills": ["skill1", "skill2", ...],
  "missing_skills": ["skill1", "skill2", ...],
  "experience_match": "description of experience alignment",
  "education_match": "description of education alignment",
  "improvement_suggestions": ["suggestion1", "suggestion2", ...],
  "key_strengths": ["strength1", "strength2", ...],
  "overall_feedback": "comprehensive feedback text"
}

Ensure all fields are present and properly formatted."""

class LangChainResumeAnalyzer:
    """LangChain-based resume analyzer with structured workflows."""

    def __init__(self, api_key: str):
        """Initialize the LangChain analyzer."""
        self.api_key = api_key
        self.llm = None
        self.chains = {}
        self.available = LANGCHAIN_AVAILABLE

        if self.available:
            self._initialize_llm()
            self._setup_chains()
        else:
            logger.warning("LangChain not available, using fallback mode")

    def _initialize_llm(self):
        """Initialize the Google Gemini LLM through LangChain."""
        if not self.available:
            return

        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0.1,
                max_tokens=2048
            )
            logger.info("LangChain LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain LLM: {e}")
            self.available = False

    def _initialize_llm(self):
        """Initialize the Google Gemini LLM through LangChain."""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0.1,
                max_tokens=2048
            )
            logger.info("LangChain LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain LLM: {e}")
            raise

    def _setup_chains(self):
        """Set up LangChain chains for different analysis tasks."""
        if not self.available:
            return

        try:
            # Main resume analysis chain
            analysis_template = """
You are an expert HR professional analyzing a resume against a job description.

RESUME CONTENT:
{resume_text}

JOB DESCRIPTION:
{job_description}

Please analyze the resume for relevance to this job position. Focus on:
1. Technical skills match
2. Experience level alignment
3. Education requirements
4. Industry background
5. Key qualifications and certifications

{format_instructions}

Return only the JSON analysis, no additional text.
"""

            analysis_prompt = PromptTemplate(
                template=analysis_template,
                input_variables=["resume_text", "job_description"],
                partial_variables={"format_instructions": ResumeAnalysisOutputParser().get_format_instructions()}
            )

            self.chains['analysis'] = LLMChain(
                llm=self.llm,
                prompt=analysis_prompt,
                output_parser=ResumeAnalysisOutputParser(),
                verbose=False
            )

            logger.info("LangChain chains initialized successfully")

        except Exception as e:
            logger.error(f"Failed to setup chains: {e}")
            self.available = False

    def analyze_resume(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze resume against job description using LangChain."""
        if not self.available:
            logger.warning("LangChain not available, returning fallback response")
            return self._get_fallback_response()

        try:
            # Run the analysis chain
            result = self.chains['analysis'].run(
                resume_text=resume_text,
                job_description=job_description
            )

            # Ensure result is a dictionary
            if isinstance(result, str):
                # If it's a string, try to parse it as JSON
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    logger.error("Failed to parse analysis result as JSON")
                    return self._get_fallback_response()

            logger.info("Resume analysis completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in resume analysis: {e}")
            return self._get_fallback_response()

    def extract_resume_skills(self, resume_text: str) -> Dict[str, List[str]]:
        """Extract skills from resume using LangChain."""
        try:
            result = self.chains['skills_extraction'].run(resume_text=resume_text)

            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    logger.error("Failed to parse skills extraction result")
                    return {}

            return result

        except Exception as e:
            logger.error(f"Error extracting skills: {e}")
            return {}

    def extract_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """Extract requirements from job description using LangChain."""
        try:
            result = self.chains['jd_extraction'].run(job_description=job_description)

            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JD extraction result")
                    return {}

            return result

        except Exception as e:
            logger.error(f"Error extracting job requirements: {e}")
            return {}

    def batch_analyze(self, resume_files, job_description: str) -> List[Dict[str, Any]]:
        """Analyze multiple resumes using LangChain."""
        results = []

        for resume_file in resume_files:
            try:
                # Extract text (assuming this is handled elsewhere)
                resume_text = self._extract_text_from_file(resume_file)

                if resume_text:
                    analysis = self.analyze_resume(resume_text, job_description)
                    analysis['filename'] = resume_file.name
                    results.append(analysis)
                else:
                    results.append(self._get_fallback_response())

            except Exception as e:
                logger.error(f"Error analyzing {resume_file.name}: {e}")
                results.append(self._get_fallback_response())

        return results

    def _extract_text_from_file(self, file) -> str:
        """Extract text from uploaded file (placeholder - implement based on your needs)."""
        # This should be implemented based on your existing text extraction logic
        # For now, return empty string
        return ""

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

    def get_chain_info(self) -> Dict[str, Any]:
        """Get information about available chains."""
        return {
            'available_chains': list(self.chains.keys()),
            'llm_model': str(self.llm.model_name) if hasattr(self.llm, 'model_name') else 'Unknown'
        }