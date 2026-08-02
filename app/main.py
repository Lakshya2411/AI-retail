import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Header, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"

from app.services.cv_service import CVService, VISIT_LOGS
from app.services.nlp_service import NLPService, SENTIMENT_LOGS
from app.services.chatbot_service import ChatbotService, CHATBOT_LOGS
from app.routers import vision, nlp, chatbot
from app.schemas import StatsResponse

API_KEY = "retail-secret-key-2026"

async def verify_api_key(x_api_key: str = Header(None, description="API Key header for authentication")):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized. Invalid or missing X-API-Key header."
        )
    return x_api_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Smart Retail Services and loading models...")
    app.state.cv_service = CVService()
    app.state.nlp_service = NLPService()
    app.state.chatbot_service = ChatbotService()
    print("Smart Retail Services initialized.")
    yield
    print("Shutting down Smart Retail Services...")

app = FastAPI(
    title="Smart Retail & Customer Intelligence API",
    description="Production-style API serving Computer Vision face recognition and product classification, NLP sentiment feedback classification, and FAQ chatbot.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")

app.include_router(vision.router, dependencies=[Depends(verify_api_key)])
app.include_router(nlp.router, dependencies=[Depends(verify_api_key)])
app.include_router(chatbot.router, dependencies=[Depends(verify_api_key)])

@app.get("/dashboard/stats", response_model=StatsResponse, tags=["Dashboard Analytics"])
async def get_dashboard_stats():
    """
    Returns aggregated visit and sentiment statistics along with historical activity logs.
    """
    total_visits = len(VISIT_LOGS)
    known_customer_visits = len([v for v in VISIT_LOGS if v["customer_id"] != -1])
    unknown_customer_visits = len([v for v in VISIT_LOGS if v["customer_id"] == -1])
    
    total_sentiments = len(SENTIMENT_LOGS)
    
    sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
    sum_confidence = 0.0
    for s in SENTIMENT_LOGS:
        sentiment_distribution[s["sentiment"]] = sentiment_distribution.get(s["sentiment"], 0) + 1
        sum_confidence += s["confidence"]
        
    avg_confidence = (sum_confidence / total_sentiments) if total_sentiments > 0 else 0.0
    
    return StatsResponse(
        total_visits=total_visits,
        known_customer_visits=known_customer_visits,
        unknown_customer_visits=unknown_customer_visits,
        total_sentiments=total_sentiments,
        sentiment_distribution=sentiment_distribution,
        avg_sentiment_confidence=round(avg_confidence, 4),
        visit_logs=list(reversed(VISIT_LOGS)),
        sentiment_logs=list(reversed(SENTIMENT_LOGS)),
        chatbot_logs=list(reversed(CHATBOT_LOGS))
    )
