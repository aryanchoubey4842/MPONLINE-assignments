# Document Knowledge Base — Local RAG Assistant

A clean, production-grade Retrieval-Augmented Generation (RAG) chatbot designed to ingest, index, and query dense PDF documents natively. The system relies on LangChain for pipeline architecture, Google Gemini for semantic embeddings and text synthesis, and ChromaDB for local vector storage.

## Key Features

* **Paced Ingestion Pipeline:** Self-throttling ingestion mechanism with automated backoff logic to safely stay within free-tier API Rate Limits (RPM).
* **Persistent Local Storage:** Contextual segments are vectorized and cached locally in an optimized Chroma DB structure to prevent redundant API calls.
* **Modern Web Interface:** A premium HTML/CSS/JS frontend with dark mode, animations, and a sleek user experience, served by a Flask backend.
* **Context Optimization:** Fine-tuned token character splitting ($1000$ chunk size, $150$ overlap) paired with an expanded retrieval window ($k=6$) to minimize text fragmentation and eliminate hallucinations.
* **Docker Ready:** Fully containerized architecture using Docker, making it easy to deploy on platforms like Render.com.

---

## Directory Architecture

```text
├── data/               # Source PDFs go here for processing
├── .chroma_db/         # Generated persistent vector store database
├── .env                # Local environmental secrets
├── .gitignore          # Git exclusion framework
├── .dockerignore       # Docker exclusion framework
├── requirements.txt    # Application dependencies
├── Dockerfile          # Containerization instructions
├── ingest.py           # Document chunking & vector ingestion pipeline
├── app.py              # Flask backend server
├── static/             # CSS styling and JS logic for the frontend
└── templates/          # HTML templates for the frontend
```

---

## Installation & Environment Configuration

### 1. Environment Secrets Setup

Create a `.env` file in the root project directory and provide your API configuration:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 2. Ingesting Target Documents (Required before running)

1. Drop your sample PDF or text documentation directly inside the `data/` directory.
2. Install the necessary packages locally (if you don't use Docker for ingestion):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the paced indexing pipeline from your terminal:
   ```bash
   python ingest.py
   ```
   The script will compute semantic embedding vectors and save them inside the local storage bucket (`.chroma_db/`).

---

## Launch Instructions

You can run this application natively or using Docker.

### Option A: Run Locally (Native Python)

1. Ensure your virtual environment is active and dependencies are installed (`pip install -r requirements.txt`).
2. Start the Flask application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to `http://localhost:5000`.

### Option B: Run with Docker (Recommended for Deployment)

1. Ensure Docker is installed and running on your system.
2. Build the Docker image:
   ```bash
   docker build -t rag-bot .
   ```
3. Run the Docker container, mapping port 5000:
   ```bash
   docker run -p 5000:5000 --env-file .env rag-bot
   ```
4. Open your browser and navigate to `http://localhost:5000`.

---

## Deploying to Render.com

1. Push your repository to GitHub (ensure you include the `.chroma_db` if you don't run ingestion on startup).
2. Go to Render.com and create a new **Web Service**.
3. Connect your repository.
4. Select **Docker** as the Runtime.
5. In the Environment Variables section, add `GEMINI_API_KEY` and set it to your actual key.
6. Deploy!
