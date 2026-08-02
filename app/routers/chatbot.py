from fastapi import APIRouter, Request, HTTPException
from app.schemas import ChatbotRequest, ChatbotResponse

router = APIRouter(
    prefix="/chatbot",
    tags=["FAQ Chatbot"]
)

@router.post("/", response_model=ChatbotResponse)
async def chatbot_chat(request: Request, body: ChatbotRequest):
    """
    Interacts with the FAQ chatbot. Processes user queries using exact rule matching
    with an ML classifier fallback, and logs conversation logs.
    """
    chatbot_service = request.app.state.chatbot_service
    
    try:
        res = chatbot_service.get_reply(body.message)
        if "status" in res and res["status"] == "error":
            raise HTTPException(status_code=500, detail=res["message"])
            
        return ChatbotResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot failed: {str(e)}")
