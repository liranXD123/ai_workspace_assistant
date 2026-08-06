# 🤖 AI Workspace & Ergonomics Assistant

An end-to-end, privacy-focused Edge AI desktop application that monitors user posture, gaze direction, and workspace distractions in real time. It delivers automated executive performance summaries using a local LLM and securely persists multi-tenant session history.

---

## 🌟 Key Features

- **Real-Time Computer Vision:** Pose estimation and phone detection powered by YOLOv8 and OpenCV.
- **Dynamic Posture & Gaze Calibration:** 3D head yaw/pitch ratio estimation and slouch detection normalized to dynamic user baselines.
- **Local LLM Executive Summaries:** Generates structured end-of-session feedback reports locally using Ollama (`llama3.2:3b`) and LangChain.
- **Multi-Tenant Security:** Full signup/login authentication using `bcrypt` password hashing and 100% parameterized SQL queries (immune to SQL Injection).
- **Session History Dashboard:** Persistent session tracking backed by SQLite with automatic high-priority expanding for recent records.
- **One-Command Orchestration:** Containerized via Docker & Docker Compose for zero-config deployment.

---

## 🏗️ Architecture & Tech Stack
[ Webcam Feed ] ──> [ YOLOv8 Pose / Object ] ──> [ OpenCV Geometry Engine ]
│
▼
[ SQLite DB ] <── [ Secure Auth / Session ] <── [ Streamlit Dashboard ]
│
▼
[ Ollama (llama3.2:3b) ]

- **Frontend / UI:** Streamlit
- **Vision Engine:** OpenCV, Ultralytics YOLOv8 (`yolov8n-pose`, `yolov8n`)
- **LLM Engine:** Ollama, LangChain
- **Backend & Database:** SQLite3, `bcrypt`
- **Containerization:** Docker, Docker Compose

---

## 🚀 Quickstart Guide
### Option 1: Running with Docker Compose (Recommended)

Make sure [Docker Desktop](https://www.docker.com/) is installed and running, then execute:

bash
docker-compose up -d


---

### 2. 🙈 `.gitignore`

Save this as `.gitignore` in your project root directory:

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Virtual Environments
.venv/
venv/
ENV/
env/

# SQLite Database & Storage
*.db
*.sqlite
*.sqlite3
database.db

# Machine Learning Model Weights
*.pt
*.onnx
*.engine

# Operating System Files
.DS_Store
Thumbs.db
ehthumbs.db

# IDEs & Code Editors
.idea/
.vscode/
*.swp
*.swo

# Session Logs & Debug Artifacts
*.log
session_log.json
workspace_logs.json

```bash
docker-compose up -d
