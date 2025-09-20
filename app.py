import streamlit as st
import os
import pandas as pd
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def is_cloud_environment():
    """Detect if running in a cloud environment (Streamlit Cloud, etc.)"""
    # Check for Streamlit Cloud specific environment variables
    cloud_indicators = [
        'STREAMLIT_SERVER_HEADLESS',  # Streamlit Cloud
        'STREAMLIT_CLOUD',           # Custom indicator
        os.getenv('STREAMLIT_RUNTIME')  # Another indicator
    ]
    return any(cloud_indicators)

# Simple fallback analyzer for now
class SimpleResumeAnalyzer:
    def __init__(self):
        # Try to get API key from Streamlit secrets first, then environment variables
        try:
            self.api_key = st.secrets.get('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY', 'demo_key'))
        except (AttributeError, KeyError):
            # Fallback for when Streamlit secrets not accessible
            self.api_key = os.getenv('GEMINI_API_KEY', 'demo_key')

        # Initialize Gemini
        if self.api_key and self.api_key != 'demo_key':
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.use_gemini = True
                print("✅ Gemini AI initialized successfully!")
            except Exception as e:
                print(f"❌ Failed to initialize Gemini: {e}")
                self.use_gemini = False
        else:
            self.use_gemini = False
            print("⚠️ No Gemini API key found, using fallback analysis")

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            import fitz  # PyMuPDF
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
            from docx import Document
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

    def get_verdict_color(self, verdict: str) -> str:
        """Get color for verdict display."""
        colors = {
            'High': '#28a745',    # Green
            'Medium': '#ffc107',  # Yellow
            'Low': '#dc3545'      # Red
        }
        return colors.get(verdict, '#6c757d')

    def batch_analyze(self, resume_files, jd_text: str, use_enhanced: bool = None) -> list:
        """Analyze multiple resumes against a single job description."""
        results = []

        progress_bar = st.progress(0)

        for i, resume_file in enumerate(resume_files):
            st.write(f"Processing: {resume_file.name}")

            # Extract resume text
            resume_text = self.extract_text_from_file(resume_file)

            if resume_text:
                # Analyze resume
                analysis = self.analyze_resume(resume_text, jd_text)
                analysis['filename'] = resume_file.name
                results.append(analysis)
            else:
                # Add failed analysis
                failed_analysis = self._get_fallback_response()
                failed_analysis['filename'] = resume_file.name
                results.append(failed_analysis)

            # Update progress
            progress_bar.progress((i + 1) / len(resume_files))

        return results

    def _get_fallback_response(self) -> dict:
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

    def analyze_resume(self, resume_text: str, job_description: str) -> dict:
        """Analyze resume against job description using Gemini AI."""
        try:
            if self.use_gemini and resume_text.strip():
                return self._analyze_with_gemini(resume_text, job_description)
            else:
                return self._analyze_with_fallback(resume_text, job_description)
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return self._get_fallback_response()

    def _analyze_with_gemini(self, resume_text: str, job_description: str) -> dict:
        """Analyze using Gemini AI."""
        prompt = f"""
You are an expert HR professional analyzing a resume against a job description. Provide a detailed analysis in JSON format.

RESUME CONTENT:
{resume_text[:4000]}  # Limit to avoid token limits

JOB DESCRIPTION:
{job_description[:2000]}

Please analyze and return a JSON response with these exact fields:
{{
    "relevance_score": <number 0-100>,
    "verdict": "<High|Medium|Low>",
    "matching_skills": ["skill1", "skill2", ...],
    "missing_skills": ["skill1", "skill2", ...],
    "experience_match": "<brief description>",
    "education_match": "<brief description>",
    "improvement_suggestions": ["suggestion1", "suggestion2", ...],
    "overall_feedback": "<comprehensive feedback>",
    "key_strengths": ["strength1", "strength2", ...]
}}

Scoring Guidelines:
- 80-100: High match - Strong alignment with job requirements
- 50-79: Medium match - Some relevant skills but gaps exist
- 0-49: Low match - Limited alignment with job requirements

Be specific about skills, experience, and qualifications. Focus on technical skills, domain expertise, and job-specific requirements.
"""

        try:
            response = self.model.generate_content(prompt)

            # Extract JSON from response
            response_text = response.text.strip()

            # Clean up response (remove markdown formatting if present)
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Parse JSON
            import json
            result = json.loads(response_text)

            # Validate required fields
            required_fields = ['relevance_score', 'verdict', 'matching_skills', 'missing_skills',
                             'experience_match', 'education_match', 'improvement_suggestions',
                             'overall_feedback']

            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Ensure score is within bounds
            result['relevance_score'] = max(0, min(100, int(result['relevance_score'])))

            # Add key_strengths if missing
            if 'key_strengths' not in result:
                result['key_strengths'] = []

            return result

        except Exception as e:
            print(f"❌ Gemini analysis failed: {e}")
            return self._analyze_with_fallback(resume_text, job_description)

    def _analyze_with_fallback(self, resume_text: str, job_description: str) -> dict:
        """Fallback analysis using basic text processing."""
        try:
            # Clean and normalize texts
            resume_lower = resume_text.lower()
            jd_lower = job_description.lower()

            # Define common skills to look for
            common_skills = [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'machine learning', 'deep learning', 'ai', 'artificial intelligence',
                'data science', 'data analysis', 'statistics', 'sql', 'mysql', 'postgresql',
                'mongodb', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git',
                'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
                'html', 'css', 'bootstrap', 'tensorflow', 'pytorch', 'pandas', 'numpy',
                'scikit-learn', 'tableau', 'power bi', 'excel', 'linux', 'windows'
            ]

            # Find matching and missing skills
            matching_skills = []
            missing_skills = []

            for skill in common_skills:
                if skill in resume_lower:
                    matching_skills.append(skill.title())
                elif skill in jd_lower:
                    missing_skills.append(skill.title())

            # Calculate relevance score based on skill matches
            if not matching_skills:
                relevance_score = 25
                verdict = 'Low'
            elif len(matching_skills) < 3:
                relevance_score = 50
                verdict = 'Medium'
            elif len(matching_skills) < 6:
                relevance_score = 75
                verdict = 'Medium'
            else:
                relevance_score = 90
                verdict = 'High'

            # Generate improvement suggestions
            improvement_suggestions = []
            if missing_skills:
                improvement_suggestions.append(f"Consider learning: {', '.join(missing_skills[:3])}")
            if len(resume_text.split()) < 200:
                improvement_suggestions.append("Add more detailed work experience descriptions")
            if 'education' not in resume_lower:
                improvement_suggestions.append("Include educational background and certifications")

            # Experience analysis
            experience_keywords = ['years', 'experience', 'senior', 'junior', 'lead', 'manager']
            experience_match = "Experience level appears adequate based on content"
            if any(keyword in jd_lower for keyword in experience_keywords):
                if any(keyword in resume_lower for keyword in experience_keywords):
                    experience_match = "Experience requirements likely met"
                else:
                    experience_match = "May need more detailed experience descriptions"

            # Education analysis
            education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college']
            education_match = "Education background appears appropriate"
            if any(keyword in jd_lower for keyword in education_keywords):
                if any(keyword in resume_lower for keyword in education_keywords):
                    education_match = "Education requirements appear to be met"
                else:
                    education_match = "Consider highlighting relevant educational background"

            return {
                'relevance_score': relevance_score,
                'verdict': verdict,
                'matching_skills': matching_skills[:8],  # Limit to top 8
                'missing_skills': missing_skills[:5],    # Limit to top 5
                'experience_match': experience_match,
                'education_match': education_match,
                'improvement_suggestions': improvement_suggestions,
                'overall_feedback': f"Resume shows {len(matching_skills)} matching skills with {verdict.lower()} relevance to the job requirements.",
                'key_strengths': matching_skills[:5] if matching_skills else ['Analysis completed']
            }

        except Exception as e:
            # Fallback in case of any errors
            return {
                'relevance_score': 0,
                'verdict': 'Low',
                'matching_skills': ['Analysis failed'],
                'missing_skills': ['Unable to process'],
                'experience_match': 'Analysis failed',
                'education_match': 'Analysis failed',
                'improvement_suggestions': ['Please try again with different files'],
                'overall_feedback': f'Analysis failed: {str(e)}',
                'key_strengths': ['Analysis failed']
            }

# Use simple analyzer for now
analyzer = SimpleResumeAnalyzer()

# Simple CSV export function
def export_results_to_csv(results):
    import io
    output = io.StringIO()
    output.write("Resume,Score,Verdict,Matching Skills,Missing Skills,Feedback\n")

    for result in results:
        resume_name = result.get('resume_name', 'Unknown')
        score = result.get('relevance_score', 0)
        verdict = result.get('verdict', 'Unknown')
        matching = ', '.join(result.get('matching_skills', []))
        missing = ', '.join(result.get('missing_skills', []))
        feedback = result.get('overall_feedback', '')

        output.write(f'"{resume_name}","{score}","{verdict}","{matching}","{missing}","{feedback}"\n')

    return output.getvalue()

# Page config
st.set_page_config(
    page_title="Resume Relevance Check System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .score-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 1rem 0;
    }

    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }

    .skills-section {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Check if running in cloud environment
    cloud_env = is_cloud_environment()

    if cloud_env:
        st.info("🌐 Running in cloud environment - using optimized settings for better performance")

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📋 Resume Relevance Check System</h1>
        <p>AI-Powered Resume Evaluation Against Job Requirements</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []

    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        # Load API key from Streamlit secrets or .env
        try:
            default_api_key = st.secrets.get('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY', ''))
        except (AttributeError, KeyError):
            default_api_key = os.getenv('GEMINI_API_KEY', '')

        # API Key input with option to override
        use_custom_key = st.checkbox("🔑 Use Custom API Key", help="Check this to enter your own API key instead of using the built-in one")

        if use_custom_key:
            api_key = st.text_input(
                "Enter Your API Key",
                type="password",
                help="Enter your own Gemini API key (supports any model)"
            )
            if not api_key:
                st.warning("⚠️ Please enter your API key or uncheck 'Use Custom API Key' to use the built-in key.")
                api_key = default_api_key
        else:
            api_key = default_api_key
            if api_key:
                st.success("✅ Using built-in API key")
            else:
                st.error("❌ No API key found. Please set GEMINI_API_KEY in .env file or use custom key option.")

        if api_key:
            try:
                st.session_state.analyzer = SimpleResumeAnalyzer()
                st.success("✅ Analyzer initialized successfully!")
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")

        st.markdown("---")

        # Mode selection
        mode = st.radio(
            "📊 Analysis Mode",
            ["Single Resume", "Batch Analysis"],
            help="Choose between analyzing one resume or multiple resumes"
        )

        st.markdown("---")

        # Job Description Upload
        st.header("📝 Job Description")
        jd_file = st.file_uploader(
            "Upload Job Description",
            type=['txt', 'pdf', 'docx'],
            help="Upload the job description file"
        )

        # Or paste JD text
        jd_text = st.text_area(
            "Or paste Job Description here:",
            height=200,
            placeholder="Paste the complete job description here..."
        )

        # Process JD
        processed_jd = ""
        if jd_file and st.session_state.analyzer:
            if jd_file.type == "text/plain":
                processed_jd = str(jd_file.read(), "utf-8")
            else:
                processed_jd = st.session_state.analyzer.extract_text_from_file(jd_file)
        elif jd_text:
            processed_jd = jd_text

        if processed_jd:
            st.success(f"✅ Job Description loaded ({len(processed_jd)} characters)")

    # Main content area
    if not st.session_state.analyzer:
        st.warning("⚠️ Please configure your Gemini Pro API key in the sidebar to continue.")
        return

    if not processed_jd:
        st.warning("⚠️ Please upload or paste a job description in the sidebar to continue.")
        return

    # Main analysis interface
    if mode == "Single Resume":
        single_resume_analysis(processed_jd)
    else:
        batch_resume_analysis(processed_jd)

    # Results section
    if st.session_state.analysis_results:
        display_results_section()

def single_resume_analysis(jd_text):
    """Handle single resume analysis."""
    st.header("📄 Single Resume Analysis")

    # Resume upload
    col1, col2 = st.columns([2, 1])

    with col1:
        resume_file = st.file_uploader(
            "Upload Resume",
            type=['pdf', 'docx'],
            help="Upload a PDF or DOCX resume file"
        )

    with col2:
        if st.button("🔍 Analyze Resume", type="primary", disabled=not resume_file):
            if resume_file:
                analyze_single_resume(resume_file, jd_text)

def batch_resume_analysis(jd_text):
    """Handle batch resume analysis."""
    st.header("📊 Batch Resume Analysis")

    # Multiple resume upload
    col1, col2 = st.columns([2, 1])

    with col1:
        resume_files = st.file_uploader(
            "Upload Multiple Resumes",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Upload multiple PDF or DOCX resume files"
        )

        if resume_files:
            st.info(f"📁 {len(resume_files)} files selected for analysis")

    with col2:
        if st.button("🔍 Analyze All Resumes", type="primary", disabled=not resume_files):
            if resume_files:
                analyze_batch_resumes(resume_files, jd_text)

def analyze_single_resume(resume_file, jd_text):
    """Analyze a single resume."""
    with st.spinner("🔍 Analyzing resume..."):
        try:
            # Extract text
            resume_text = st.session_state.analyzer.extract_text_from_file(resume_file)

            if not resume_text:
                st.error("❌ Failed to extract text from resume. Please check the file format.")
                return

            # Analyze
            result = st.session_state.analyzer.analyze_resume(resume_text, jd_text)
            result['filename'] = resume_file.name
            result['analysis_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Store result
            st.session_state.analysis_results = [result]  # Replace previous single result

            st.success("✅ Analysis completed!")

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

def analyze_batch_resumes(resume_files, jd_text):
    """Analyze multiple resumes."""
    with st.spinner("🔍 Analyzing resumes..."):
        try:
            results = st.session_state.analyzer.batch_analyze(resume_files, jd_text)

            # Add timestamp to results
            for result in results:
                result['analysis_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Store results
            st.session_state.analysis_results = results

            st.success(f"✅ Batch analysis completed! Processed {len(results)} resumes.")

        except Exception as e:
            st.error(f"❌ Batch analysis failed: {str(e)}")

def display_results_section():
    """Display analysis results."""
    st.markdown("---")
    st.header("📊 Analysis Results")

    results = st.session_state.analysis_results

    if len(results) == 1:
        # Single result display
        display_single_result(results[0])
    else:
        # Batch results display
        display_batch_results(results)

def display_single_result(result):
    """Display single resume analysis result."""
    filename = result.get('filename', 'Unknown')
    score = result.get('relevance_score', 0)
    verdict = result.get('verdict', 'Low')

    # Main score display
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        score_color = st.session_state.analyzer.get_score_color(score)
        verdict_color = st.session_state.analyzer.get_verdict_color(verdict)

        st.markdown(f"""
        <div class="score-card">
            <h2 style="color: {score_color}; margin: 0;">{score}/100</h2>
            <h3 style="color: {verdict_color}; margin: 10px 0;">Relevance Score</h3>
            <span style="background: {verdict_color}; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;">
                {verdict} Match
            </span>
            <p style="margin-top: 15px; color: #666; font-size: 14px;">📄 {filename}</p>
        </div>
        """, unsafe_allow_html=True)

    # Detailed analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Matching Skills")
        matching_skills = result.get('matching_skills', [])
        if matching_skills:
            for skill in matching_skills[:10]:
                st.markdown(f"✅ **{skill}**")
        else:
            st.info("No matching skills identified")

        st.markdown("### 📚 Education Match")
        st.info(result.get('education_match', 'Not analyzed'))

    with col2:
        st.markdown("### ❌ Missing Skills")
        missing_skills = result.get('missing_skills', [])
        if missing_skills:
            for skill in missing_skills[:10]:
                st.markdown(f"❌ **{skill}**")
        else:
            st.success("No significant skills missing")

        st.markdown("### 💼 Experience Match")
        st.info(result.get('experience_match', 'Not analyzed'))

    # Improvement suggestions
    st.markdown("### 💡 Improvement Suggestions")
    suggestions = result.get('improvement_suggestions', [])
    if suggestions:
        for i, suggestion in enumerate(suggestions[:5], 1):
            st.markdown(f"**{i}.** {suggestion}")
    else:
        st.info("No specific suggestions available")

    # Overall feedback
    st.markdown("### 📋 Overall Feedback")
    feedback = result.get('overall_feedback', 'No feedback available')
    st.markdown(f"*{feedback}*")

    # Key strengths
    if 'key_strengths' in result and result['key_strengths']:
        st.markdown("### 🌟 Key Strengths")
        for strength in result['key_strengths'][:5]:
            st.markdown(f"🌟 **{strength}**")

def display_batch_results(results):
    """Display batch analysis results."""
    # Summary statistics
    st.markdown("### 📈 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    total_resumes = len(results)
    high_match = len([r for r in results if r.get('verdict') == 'High'])
    medium_match = len([r for r in results if r.get('verdict') == 'Medium'])
    low_match = len([r for r in results if r.get('verdict') == 'Low'])
    avg_score = sum([r.get('relevance_score', 0) for r in results]) / total_resumes if total_resumes > 0 else 0

    with col1:
        st.metric("Total Resumes", total_resumes)
    with col2:
        st.metric("High Match", high_match, f"{(high_match/total_resumes*100):.1f}%")
    with col3:
        st.metric("Medium Match", medium_match, f"{(medium_match/total_resumes*100):.1f}%")
    with col4:
        st.metric("Average Score", f"{avg_score:.1f}", f"Low: {low_match}")

    # Results table
    st.markdown("### 📊 Detailed Results")

    # Create dataframe for display
    df_data = []
    for result in results:
        df_data.append({
            'Filename': result.get('filename', 'Unknown')[:30] + '...' if len(result.get('filename', '')) > 30 else result.get('filename', 'Unknown'),
            'Score': result.get('relevance_score', 0),
            'Verdict': result.get('verdict', 'Low'),
            'Top Skills': ', '.join(result.get('matching_skills', [])[:3]),
            'Missing Skills': ', '.join(result.get('missing_skills', [])[:3]),
            'Analysis Time': result.get('analysis_time', 'Unknown')
        })

    df = pd.DataFrame(df_data)

    # Color code the dataframe
    def color_verdict(val):
        if val == 'High':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Medium':
            return 'background-color: #fff3cd; color: #856404'
        else:
            return 'background-color: #f8d7da; color: #721c24'

    def color_score(val):
        if val >= 80:
            return 'background-color: #d4edda; color: #155724; font-weight: bold'
        elif val >= 60:
            return 'background-color: #fff3cd; color: #856404; font-weight: bold'
        elif val >= 40:
            return 'background-color: #ffeaa7; color: #6c5ce7; font-weight: bold'
        else:
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold'

    styled_df = df.style.applymap(color_verdict, subset=['Verdict']).applymap(color_score, subset=['Score'])
    st.dataframe(styled_df, use_container_width=True)

    # Export options
    st.markdown("### 📥 Export Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        # CSV export
        csv_data = export_results_to_csv(results)
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        # JSON export
        json_data = json.dumps(results, indent=2)
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col3:
        # Summary report
        summary_report = generate_summary_report(results)
        st.download_button(
            label="📄 Summary Report",
            data=summary_report,
            file_name=f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    # Individual result viewer
    st.markdown("### 🔍 Individual Result Viewer")

    if results:
        selected_resume = st.selectbox(
            "Select resume to view details:",
            options=range(len(results)),
            format_func=lambda x: f"{results[x].get('filename', 'Unknown')} (Score: {results[x].get('relevance_score', 0)})"
        )

        if st.button("View Details"):
            st.markdown("---")
            st.markdown("#### Detailed Analysis")
            display_single_result(results[selected_resume])

def generate_summary_report(results):
    """Generate a text summary report."""
    total_resumes = len(results)
    high_match = len([r for r in results if r.get('verdict') == 'High'])
    medium_match = len([r for r in results if r.get('verdict') == 'Medium'])
    low_match = len([r for r in results if r.get('verdict') == 'Low'])
    avg_score = sum([r.get('relevance_score', 0) for r in results]) / total_resumes if total_resumes > 0 else 0

    # Get top candidates
    sorted_results = sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
    top_5 = sorted_results[:5]

    # Common missing skills
    all_missing_skills = []
    for result in results:
        all_missing_skills.extend(result.get('missing_skills', []))

    from collections import Counter
    common_missing = Counter(all_missing_skills).most_common(10)

    report = f"""
RESUME ANALYSIS SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== OVERVIEW ===
Total Resumes Analyzed: {total_resumes}
Average Relevance Score: {avg_score:.1f}/100

=== DISTRIBUTION ===
High Match (80-100):   {high_match} ({high_match/total_resumes*100:.1f}%)
Medium Match (50-79):  {medium_match} ({medium_match/total_resumes*100:.1f}%)
Low Match (0-49):      {low_match} ({low_match/total_resumes*100:.1f}%)

=== TOP 5 CANDIDATES ===
"""

    for i, candidate in enumerate(top_5, 1):
        report += f"{i}. {candidate.get('filename', 'Unknown')} - Score: {candidate.get('relevance_score', 0)}/100 ({candidate.get('verdict', 'Low')})\n"

    report += f"""
=== COMMON SKILL GAPS ===
"""

    for skill, count in common_missing:
        report += f"- {skill}: Missing in {count} resumes ({count/total_resumes*100:.1f}%)\n"

    report += f"""
=== RECOMMENDATIONS ===
1. Focus recruitment on High Match candidates ({high_match} candidates available)
2. Consider skill development programs for top skill gaps
3. Review job requirements if overall match rates are low
4. Interview top {min(10, high_match + medium_match)} candidates based on scores

=== DETAILED RESULTS ===
"""

    for result in sorted_results:
        report += f"""
Filename: {result.get('filename', 'Unknown')}
Score: {result.get('relevance_score', 0)}/100
Verdict: {result.get('verdict', 'Low')}
Key Strengths: {', '.join(result.get('key_strengths', ['None'])[:3])}
Missing Skills: {', '.join(result.get('missing_skills', ['None'])[:3])}
---
"""

    return report

def main_app():
    """Main application entry point."""
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main_app()