# 📊 Data Analyst Agent

An autonomous AI-powered Data Analyst Agent built using FastAPI, LangChain, Gemini LLMs, Pandas, and Python sandboxed execution.

This project allows users to:
- Upload datasets
- Ask analytical questions
- Perform autonomous data analysis
- Generate plots
- Scrape web data
- Execute AI-generated Python safely
- Return structured JSON responses

---

🌐 **Live App:** [Data Analyst Agent](https://data-analyst-agent-cl2b.onrender.com/)

## 🚀 Features

### ✅ AI-Powered Data Analysis
- Gemini-powered autonomous reasoning
- Multi-key fallback system
- Model hierarchy support
- Intelligent dataset analysis

### ✅ Dataset Support

Supports: CSV, Excel (`.xlsx`, `.xls`), JSON, Parquet, Images (`.png`, `.jpg`, `.jpeg`)

### ✅ Autonomous Python Execution

The AI agent:
1. Reads uploaded datasets
2. Understands analytical questions
3. Generates Python code dynamically
4. Executes code safely in a sandbox
5. Returns structured results

### ✅ Smart Web Scraping

Built-in scraping pipeline:
- HTML tables
- CSV links
- JSON APIs
- Plain text fallback

### ✅ Visualization Support

Supports Matplotlib and Seaborn with Base64 plot generation.

### ✅ Production Ready

- FastAPI backend
- Dockerized deployment
- Render compatible
- Timeout protection
- Health diagnostics endpoint
- Multi-LLM retry system
- Dataset memory protection

---

## 🏗️ Tech Stack

| Technology | Purpose |
|---|---|
| FastAPI | Backend API |
| LangChain | Agent orchestration |
| Gemini API | LLM |
| Pandas | Data analysis |
| NumPy | Numerical computing |
| Matplotlib | Plot generation |
| Seaborn | Visualization |
| BeautifulSoup | Web scraping |
| Docker | Containerization |
| Render | Deployment |

---

## 📂 Project Structure

```bash
project-root/
│
├── app.py              # Main FastAPI application
├── index.html          # Frontend UI
├── Dockerfile          # Docker configuration
├── Procfile            # Render process configuration
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python runtime version
├── entrypoint.sh       # Startup script
├── .gitignore          # Git ignored files
└── README.md           # Project documentation
```

---

## ⚙️ Environment Variables

Create a `.env` file in the root directory:

```env
# Gemini API Keys
gemini_api_1=YOUR_API_KEY
gemini_api_2=YOUR_API_KEY
gemini_api_3=YOUR_API_KEY
gemini_api_4=YOUR_API_KEY
gemini_api_5=YOUR_API_KEY

# LLM Timeout
LLM_TIMEOUT_SECONDS=240
```

### 🔑 Getting Gemini API Keys

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Create API Keys
3. Add them to `.env`

---

## 🐳 Docker Setup

**Build Docker Image**
```bash
docker build -t tds-agent .
```

**Run Docker Container**
```bash
docker run -p 8000:8000 tds-agent
```

---

## ▶️ Local Development Setup

**1. Clone Repository**
```bash
git clone <repo_url>
cd project-folder
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Run Application**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 🌐 API Endpoints

### `GET /`
Serves the frontend UI.

### `GET /api`
Health check endpoint.

**Example Response**
```json
{
  "ok": true,
  "service": "TDS Data Analyst Agent",
  "status": "running"
}
```

### `POST /api`
Main analysis endpoint.

**Upload:**
- Questions file (`.txt`)
- Optional dataset

**Returns:**
```json
{
  "answer1": "...",
  "answer2": "..."
}
```

### `GET /summary`
Advanced diagnostics endpoint. Checks:
- Environment
- Gemini keys
- System memory
- Disk space
- Pandas pipeline
- Network connectivity

---

## 🧠 LLM Architecture

### Gemini Fallback System

The system:
- Rotates API keys
- Uses model hierarchy
- Automatically retries failed keys
- Prevents quota failures

### Model Hierarchy

```python
MODEL_HIERARCHY = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite"
]
```

---

## 🔒 Security Features

### Sandbox Execution

LLM-generated code runs inside:
- Temporary isolated scripts
- Restricted execution environment
- Timeout-controlled execution

### Unsafe Code Protection

Blocks: `os`, `subprocess`, `open()`, `sys.exit`

---

## 📊 Plot Handling

Plots are:
- Generated using Matplotlib
- Converted to Base64
- Size-limited for API responses

---

## 🕷️ Scraping Pipeline

The scraping tool downloads a webpage, detects the content type, extracts tables, and falls back to plain text.

Supports: CSV, HTML, JSON, Excel

---

## 🧪 Example Usage

**Example Questions File**
```
- `q1`: string
What is the average sales amount?

- `q2`: number
What is the maximum revenue?
```

**Example Response**
```json
{
  "q1": "The average sales amount is 1450.23",
  "q2": 9821
}
```

---

## 🚀 Render Deployment Guide

**Step 1 — Push Project to GitHub**

Push all project files to GitHub.

**Step 2 — Create Render Web Service**
1. Open [Render](https://render.com)
2. Click **New Web Service**
3. Connect your GitHub repository
4. Select branch: `main`
5. Environment: **Docker**

**Step 3 — Add Environment Variables**

Add the following:
- `gemini_api_1`
- `gemini_api_2`
- `gemini_api_3`
- `LLM_TIMEOUT_SECONDS`

**Step 4 — Deploy**

Render automatically detects the Dockerfile, builds the image, and starts the FastAPI server.

---

## ⚡ Performance Features

- Multi-key Gemini fallback
- Retry logic
- Timeout handling
- Dataset truncation
- Payload limiting
- Safe execution environment

---

## 📋 Supported File Formats

| File Type | Supported |
|---|---|
| CSV | ✅ |
| Excel | ✅ |
| JSON | ✅ |
| Parquet | ✅ |
| PNG | ✅ |
| JPG | ✅ |
| JPEG | ✅ |

---

## 📦 Main Dependencies

```
fastapi
uvicorn[standard]
python-multipart
pandas
numpy
matplotlib
seaborn
networkx
requests
python-dotenv
beautifulsoup4
lxml
pillow
langchain
langchain-core
langchain-google-genai
google-generativeai
openpyxl
pyarrow
tabulate
html5lib
duckdb
psutil
httpx
scikit-learn
```

---

## 🧪 Diagnostics & Monitoring

The application includes:
- Health monitoring
- Network diagnostics
- Gemini API testing
- System resource checks
- Pandas execution validation

---

## 🔄 Agent Workflow

```
User Uploads Dataset
        ↓
Questions Parsed
        ↓
LLM Generates Python Code
        ↓
Sandbox Execution
        ↓
Results Extracted
        ↓
JSON Response Returned
```

---

## 📈 Future Improvements

- Multi-provider routing
- OpenAI integration
- Streaming responses
- Vector database support
- Distributed execution
- Async execution engine
- Advanced caching

---

## 👨‍💻 Author

Developed by **Pratyush Nishank**
Portfolio: https://pratyush-nishank-portfolio.vercel.app

---

## 📄 License

MIT License
