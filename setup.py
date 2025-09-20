import os
import subprocess
import sys

def create_project_structure():
    """Create the basic project structure."""
    directories = [
        'sample_data',
        'sample_data/sample_resumes', 
        'sample_data/sample_jds',
        '.streamlit',
        'assets',
        'assets/screenshots'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_streamlit_secrets():
    """Create secrets.toml template."""
    secrets_content = '''# Copy your API key here
GEMINI_API_KEY = "your_gemini_api_key_here"
'''
    
    with open('.streamlit/secrets.toml', 'w') as f:
        f.write(secrets_content)
    print("✅ Created .streamlit/secrets.toml template")

def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/

# Streamlit
.streamlit/secrets.toml

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary files
*.tmp
*.temp
'''
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✅ Created .gitignore file")

def create_sample_job_description():
    """Create a sample job description for testing."""
    jd_content = '''Data Scientist - AI/ML Engineer

Company: Tech Innovation Labs
Location: Bangalore, India

Job Description:
We are seeking an experienced Data Scientist with expertise in Machine Learning and Artificial Intelligence to join our growing team.

Required Skills:
- Python programming (3+ years experience)
- Machine Learning frameworks: Scikit-learn, TensorFlow, PyTorch
- Data manipulation: Pandas, NumPy
- Database skills: SQL, NoSQL
- Statistical analysis and data visualization
- Experience with cloud platforms (AWS, Azure, GCP)

Preferred Skills:
- Deep Learning and Neural Networks
- Natural Language Processing
- Computer Vision
- MLOps and model deployment
- Docker and Kubernetes
- Big data technologies: Spark, Hadoop

Education:
- Bachelor's or Master's degree in Computer Science, Statistics, Mathematics, or related field

Experience:
- 3-5 years of experience in data science or machine learning roles
- Experience with end-to-end ML project lifecycle
- Strong problem-solving and analytical thinking skills

Responsibilities:
- Develop and deploy machine learning models
- Analyze large datasets to extract meaningful insights
- Collaborate with cross-functional teams
- Present findings to stakeholders
- Stay updated with latest AI/ML trends and technologies
'''
    
    with open('sample_data/sample_jds/data_scientist_jd.txt', 'w') as f:
        f.write(jd_content)
    print("✅ Created sample job description")

def install_requirements():
    """Install required packages."""
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please run: pip install -r requirements.txt")

def create_run_script():
    """Create a simple run script."""
    if os.name == 'nt':  # Windows
        run_content = '''@echo off
echo Starting Resume Relevance Check System...
streamlit run app.py
pause
'''
        with open('run.bat', 'w') as f:
            f.write(run_content)
        print("✅ Created run.bat for Windows")
    else:  # Unix/Linux/Mac
        run_content = '''#!/bin/bash
echo "Starting Resume Relevance Check System..."
streamlit run app.py
'''
        with open('run.sh', 'w') as f:
            f.write(run_content)
        os.chmod('run.sh', 0o755)
        print("✅ Created run.sh for Unix/Linux/Mac")

def main():
    """Main setup function."""
    print("🚀 Setting up Resume Relevance Check System for Code4Edtech Hackathon")
    print("=" * 60)
    
    # Create project structure
    create_project_structure()
    
    # Create configuration files
    create_streamlit_secrets()
    create_gitignore()
    
    # Create sample data
    create_sample_job_description()
    
    # Create run script
    create_run_script()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Add your Gemini API key to .streamlit/secrets.toml")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run: streamlit run app.py")
    print("\n🔗 Useful Commands:")
    print("- Start app: streamlit run app.py")
    print("- Install deps: pip install -r requirements.txt")
    print("- Create venv: python -m venv venv")
    print("- Activate venv: source venv/bin/activate (Unix) or venv\\Scripts\\activate (Windows)")
    print("\n🎯 Hackathon Success Tips:")
    print("- Test with multiple resume formats")
    print("- Prepare impressive demo scenarios")  
    print("- Document your code thoroughly")
    print("- Create engaging video demonstration")
    print("\n🏆 Good luck with your hackathon submission!")

if __name__ == "__main__":
    main()