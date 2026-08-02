from fastapi import APIRouter, Request, HTTPException
from app.schemas import SentimentRequest, SentimentResponse

router = APIRouter(
    prefix="/nlp",
    tags=["Natural Language Processing"]
)

@router.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: Request, body: SentimentRequest):
    """
    Cleans incoming customer feedback text, runs TF-IDF vectorizer, predicts sentiment, and logs feedback metrics.
    """
    nlp_service = request.app.state.nlp_service
    if nlp_service.model is None:
        raise HTTPException(status_code=500, detail="Sentiment analysis model is offline.")
        
    try:
        res = nlp_service.analyze_sentiment(body.text)
        if "status" in res and res["status"] == "error":
            raise HTTPException(status_code=500, detail=res["message"])
            
        return SentimentResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")
