@echo off
echo Starting Resume Relevance Check System...
cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run app.py
pause