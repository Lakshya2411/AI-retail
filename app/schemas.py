from pydantic import BaseModel, Field
from typing import Dict, List, Optional

# Vision Response Schemas
class FaceRecognitionResponse(BaseModel):
    customer_id: int = Field(..., description="ID of recognized customer (-1 for unknown)")
    name: str = Field(..., description="Name of customer")
    loyalty_points: Optional[int] = Field(None, description="Loyalty program points balance")
    last_visit: Optional[str] = Field(None, description="Timestamp of last visit")
    confidence_score: float = Field(..., description="LBPH recognition confidence score (lower is more confident)")
    recognized: bool = Field(..., description="Whether the face was recognized successfully")
    message: Optional[str] = Field(None, description="Status or error message")

class ProductClassificationResponse(BaseModel):
    predicted_category: str = Field(..., description="Predicted category (shoes, bags, etc.)")
    confidence_score: float = Field(..., description="Prediction confidence score between 0 and 1")
    category_probabilities: Dict[str, float] = Field(..., description="Probability distribution across all product categories")

class Coordinate(BaseModel):
    x: int
    y: int
    w: int
    h: int

class CVOpsResponse(BaseModel):
    num_faces: int = Field(..., description="Number of faces detected")
    faces_coordinates: List[Coordinate] = Field(..., description="Bounding box coordinates of detected faces")
    gray_image_base64: str = Field(..., description="Base64 encoded grayscale image")
    edges_image_base64: str = Field(..., description="Base64 encoded Canny edges image")
    bbox_image_base64: str = Field(..., description="Base64 encoded image with face bounding boxes")

# NLP Request / Response Schemas
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer review/feedback text")

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Classified sentiment (positive, negative, neutral)")
    confidence: float = Field(..., description="Classification probability score")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution across sentiment classes")

# Chatbot Request / Response Schemas
class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User query message")

class ChatbotResponse(BaseModel):
    reply: str = Field(..., description="Chatbot generated response text")
    intent: str = Field(..., description="Matched intent tag name")
    confidence: float = Field(..., description="Intent classifier probability score")
    strategy: str = Field(..., description="Matching strategy used (rule, ml, or fallback)")

# Dashboard Stats Schemas
class VisitLogItem(BaseModel):
    timestamp: str
    customer_id: int
    name: str
    status: str

class SentimentLogItem(BaseModel):
    timestamp: str
    text: str
    sentiment: str
    confidence: float

class ChatbotLogItem(BaseModel):
    timestamp: str
    query: str
    response: str
    intent: str

class StatsResponse(BaseModel):
    total_visits: int
    known_customer_visits: int
    unknown_customer_visits: int
    total_sentiments: int
    sentiment_distribution: Dict[str, int]
    avg_sentiment_confidence: float
    visit_logs: List[VisitLogItem]
    sentiment_logs: List[SentimentLogItem]
    chatbot_logs: List[ChatbotLogItem]
