# Resume Relevance Check System - Deployment Guide
# ================================================

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)

### 1. Environment Setup
```bash
# Clone or navigate to project directory
cd "C:\Users\saket\Desktop\Resume Relevance Check System"

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
Create `.env` file in project root:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Run Application
```bash
# Start the application
streamlit run app.py --server.port=8502

# Access at: http://localhost:8502
```

---

## ☁️ Cloud Deployment Options

### Option 1: Streamlit Cloud (Easiest)
1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   # Push to your GitHub repository
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file path: `app.py`
   - Add secrets: `GEMINI_API_KEY`

### Option 2: Heroku Deployment
1. **Create requirements.txt** (already done)
2. **Create Procfile**
   ```
   web: streamlit run app.py --server.port=$PORT --server.headless=true
   ```

3. **Create runtime.txt**
   ```
   python-3.11.5
   ```

4. **Deploy**
   ```bash
   heroku create your-app-name
   heroku config:set GEMINI_API_KEY=your_key_here
   git push heroku main
   ```

### Option 3: AWS EC2
1. **Launch EC2 instance** (Ubuntu)
2. **Install dependencies**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   export GEMINI_API_KEY=your_key_here
   export STREAMLIT_SERVER_PORT=8501
   export STREAMLIT_SERVER_HEADLESS=true
   ```

4. **Run with PM2**
   ```bash
   npm install -g pm2
   pm2 start "streamlit run app.py" --name "resume-checker"
   ```

### Option 4: Docker Deployment
1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   ENV GEMINI_API_KEY=""
   ENV STREAMLIT_SERVER_PORT=8501
   ENV STREAMLIT_SERVER_HEADLESS=true

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py"]
   ```

2. **Build and run**
   ```bash
   docker build -t resume-checker .
   docker run -p 8501:8501 -e GEMINI_API_KEY=your_key resume-checker
   ```

---

## 🔧 Configuration Options

### Environment Variables
```env
# Required
GEMINI_API_KEY=your_gemini_api_key

# Optional Streamlit configs
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Database (if using external)
DATABASE_URL=sqlite:///resume_analysis.db
```

### Streamlit Configuration
Create `~/.streamlit/config.toml`:
```toml
[server]
port = 8501
headless = true
enableCORS = false

[browser]
gatherUsageStats = false
```

---

## 📊 Production Considerations

### 1. Database Setup
- Use PostgreSQL for production instead of SQLite
- Set up database migrations
- Configure connection pooling

### 2. Security
- Store API keys securely (environment variables)
- Implement authentication if needed
- Set up HTTPS/SSL certificates
- Configure CORS properly

### 3. Performance
- Implement caching for frequent queries
- Set up monitoring and logging
- Configure rate limiting for API calls
- Optimize vector search performance

### 4. Monitoring
- Set up application logging
- Monitor API usage and costs
- Track user analytics
- Set up error reporting

---

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process using port
   netstat -ano | findstr :8501
   # Kill process
   taskkill /PID <PID> /F
   ```

2. **Import errors**
   ```bash
   # Reinstall dependencies
   pip uninstall -r requirements.txt
   pip install -r requirements.txt
   ```

3. **Memory issues**
   - Reduce batch size for large file processing
   - Implement file cleanup
   - Use streaming for large files

4. **API rate limits**
   - Implement retry logic
   - Add rate limiting
   - Cache frequent requests

---

## 📈 Scaling Strategies

### Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use Redis for session management
- Implement database read replicas

### Vertical Scaling
- Increase server resources (CPU, RAM)
- Optimize database queries
- Use CDN for static assets

### Cost Optimization
- Monitor API usage costs
- Implement caching layers
- Use spot instances for development

---

## 🎯 Deployment Checklist

- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] Database initialized
- [ ] API keys secured
- [ ] Ports configured
- [ ] SSL certificates (production)
- [ ] Monitoring set up
- [ ] Backup strategy implemented
- [ ] Documentation updated

---

## 📞 Support

For deployment issues:
1. Check application logs
2. Verify environment variables
3. Test API connectivity
4. Review server resources
5. Check network configuration

**Your system is production-ready! 🚀**