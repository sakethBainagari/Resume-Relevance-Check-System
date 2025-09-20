# PowerShell script to run the Resume Relevance Check System
Write-Host "Starting Resume Relevance Check System..." -ForegroundColor Green

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Run the Streamlit app
streamlit run app.py