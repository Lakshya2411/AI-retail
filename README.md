# AI-Powered Smart Retail & Customer Intelligence Platform

An end-to-end deployed capstone platform designed for retail and e-commerce businesses. The platform integrates **Computer Vision** (face recognition for returning loyalty customers & deep learning product category classification), **Natural Language Processing** (customer review sentiment classification), and an **FAQ Chatbot** (rule-based and machine learning hybrid model) served via a production-style **FastAPI gateway** and controlled via an interactive **Streamlit dashboard**.

---

## 1. System Architecture

The platform follows a layered, service-oriented architecture:

```
                  ┌──────────────────────────────────────────────┐
                  │                 Client Layer                 │
                  │ (Streamlit Dashboard / API Swagger Docs / UI)│
                  └──────────────────────┬───────────────────────┘
                                         │
                                         │ REST Calls (HTTP POST/GET)
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │             FastAPI Gateway API              │
                  │   (/recognize-face, /classify-product, etc.) │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
     ┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────────┐
     │   Computer Vision     │ │   NLP Sentiment   │ │     Hybrid Chatbot    │
     │        Module         │ │      Module       │ │        Module         │
     │                       │ │                   │ │                       │
     │ • Haar Cascade Detect │ │ • Clean Tokens    │ │ • Exact Intent Match  │
     │ • LBPH Face Recog.    │ │ • TF-IDF Vector   │ │ • TF-IDF Fallback     │
     │ • MobileNetV2 Classify│ │ • Logistic Reg.   │ │ • Response Templates  │
     └───────────┬───────────┘ └─────────┬─────────┘ └───────────┬───────────┘
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │                Storage Layer                 │
                  │   (pkl database, xml recognizer, pt weights) │
                  └──────────────────────────────────────────────┘
```

---

## 2. Project Folder Structure

The project conforms to clean-code layout practices:

```
smart-retail-ai/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions CI/CD Pipeline
├── app/
│   ├── main.py                # FastAPI gateway entrypoint & statistics endpoint
│   ├── schemas.py             # Pydantic validation models
│   ├── dashboard.py           # Streamlit dashboard interface
│   ├── routers/
│   │   ├── vision.py          # Vision endpoints (/recognize-face, /classify-product)
│   │   ├── nlp.py             # NLP endpoints (/analyze-sentiment)
│   │   └── chatbot.py         # Chatbot endpoints (/chatbot)
│   ├── services/
│   │   ├── cv_service.py      # OpenCV operations & PyTorch classifier executor
│   │   ├── nlp_service.py     # NLTK token cleaning & sentiment model executor
│   │   └── chatbot_service.py # Conversational FAQ bot engine
│   └── models/                # Trained weights and database serializations
│       ├── train_sentiment_model.py
│       ├── train_chatbot_model.py
│       ├── train_face_recognizer.py
│       └── train_product_classifier.py
├── data/
│   ├── intents.json           # FAQ patterns and template replies
│   └── reviews.csv            # Customer feedback reviews training corpus
├── notebooks/                 # Academic exploration & model validation
│   ├── 01_image_classifier_training.ipynb
│   ├── 02_face_recognition_setup.ipynb
│   └── 03_sentiment_model_training.ipynb
├── tests/
│   └── test_endpoints.py      # pytest endpoint validation suite
├── Dockerfile                 # Containerized Docker configuration
├── requirements.txt           # Python environment packages listing
└── README.md                  # System manual and ethics documentation
```

---

## 3. Installation & Getting Started

### Prerequisites
- Python 3.11, 3.12, or 3.13 (Python 3.13 is fully tested and supported)
- Git (optional)

### Setup Instructions

1. **Clone or copy the project files** into your workspace directory.

2. **Create and Activate a Virtual Environment**:
   ```bash
   # Windows PowerShell
   py -3.13 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # Starlette test client requirements:
   pip install httpx httpx2 pytest pytest-cov
   ```

4. **Generate Datasets and Train Models**:
   Run the preprocessing and training scripts programmatically to build the serialized weights and configurations:
   ```bash
   # Generates reviews.csv
   python data/generate_synthetic_data.py
   
   # Trains models and generates .pkl, .pt, and .xml model files
   python app/models/train_sentiment_model.py
   python app/models/train_chatbot_model.py
   python app/models/train_face_recognizer.py
   python app/models/train_product_classifier.py
   ```

---

## 4. Running the Platform

### Step 1: Run the FastAPI API Gateway
Start the FastAPI server using `uvicorn`:
```bash
uvicorn app.main:app --reload
```
- **API Documentation**: Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to access the auto-generated Swagger UI.
- **API Key Security**: The vision, NLP, and chatbot endpoints require the header `X-API-Key: retail-secret-key-2026`.

### Step 2: Run the Streamlit Dashboard
Launch the dashboard in a separate shell window:
```bash
streamlit run app/dashboard.py
```
- The dashboard will open automatically at [http://localhost:8501](http://localhost:8501).
- **Dual-Mode Active**: The dashboard checks if the FastAPI REST API is online. If online, it communicates securely via HTTP calls using the secret API Key. If offline, it operates in *Local Fallback Mode* by loading the services directly into memory (useful for isolated runs).

---

## 5. Running Automated Tests

Run the test suite using `pytest` to verify Pydantic input models, endpoint permissions, and service responses:
```bash
# Disable NLTK security verification during local test suite execution
$env:NLTK_DISABLE_IMPORT_SECURITY="1"
python -m pytest tests/ -v
```

---

## 6. Deployment Configuration

### Docker Deployment
The containerized Docker build is multi-stage and optimized to support either the API Gateway or the Streamlit Dashboard.

1. **Build the Container Image**:
   ```bash
   docker build -t smart-retail-ai .
   ```
2. **Run the API Gateway Container**:
   ```bash
   docker run -d -p 8000:8000 smart-retail-ai
   ```
3. **Run the Streamlit Dashboard Container**:
   ```bash
   docker run -d -p 8501:8501 smart-retail-ai streamlit run app/dashboard.py --server.port=8501 --server.address=0.0.0.0
   ```

---

