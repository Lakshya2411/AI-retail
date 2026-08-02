import base64
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from app.schemas import FaceRecognitionResponse, ProductClassificationResponse, CVOpsResponse

router = APIRouter(
    prefix="/vision",
    tags=["Computer Vision"]
)

@router.post("/preprocess-image", response_model=CVOpsResponse)
async def preprocess_image(request: Request, file: UploadFile = File(...)):
    """
    Applies image preprocessing pipeline (Grayscale, Resize, Blur, Canny Edge Detection, Haar Cascade faces).
    Returns the preprocessed results as base64 strings.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        content = await file.read()
        cv_service = request.app.state.cv_service
        res = cv_service.apply_cv_ops(content)
        
        gray_b64 = base64.b64encode(res["gray_bytes"]).decode('utf-8')
        edges_b64 = base64.b64encode(res["edges_bytes"]).decode('utf-8')
        bbox_b64 = base64.b64encode(res["bbox_bytes"]).decode('utf-8')
        
        return CVOpsResponse(
            num_faces=res["num_faces"],
            faces_coordinates=res["faces_coordinates"],
            gray_image_base64=gray_b64,
            edges_image_base64=edges_b64,
            bbox_image_base64=bbox_b64
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image preprocessing failed: {str(e)}")

@router.post("/recognize-face", response_model=FaceRecognitionResponse)
async def recognize_face(request: Request, file: UploadFile = File(...)):
    """
    Detects face in upload, extracts LBPH features, recognizes customer, logs visit, and returns loyalty data.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        content = await file.read()
        cv_service = request.app.state.cv_service
        res = cv_service.recognize_customer(content)
        
        if "status" in res and res["status"] in ["error", "no_face_detected"]:
            return FaceRecognitionResponse(
                customer_id=-1,
                name="Unknown",
                confidence_score=0.0,
                recognized=False,
                message=res["message"]
            )
            
        return FaceRecognitionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {str(e)}")

@router.post("/classify-product", response_model=ProductClassificationResponse)
async def classify_product(request: Request, file: UploadFile = File(...)):
    """
    Predicts product category (shoes, bags, electronics, clothing, groceries) from product image using MobileNetV2.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        content = await file.read()
        cv_service = request.app.state.cv_service
        res = cv_service.classify_product(content)
        
        if "status" in res and res["status"] == "error":
            raise HTTPException(status_code=500, detail=res["message"])
            
        return ProductClassificationResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Product classification failed: {str(e)}")

@router.post("/register-face", response_model=FaceRecognitionResponse)
async def register_face(request: Request, file: UploadFile = File(...), name: str = Form(...)):
    """
    Registers a new customer face, updates the customer database, and retrains the LBPH model.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        content = await file.read()
        cv_service = request.app.state.cv_service
        res = cv_service.register_customer(content, name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
        
    if "status" in res and res["status"] == "error":
        raise HTTPException(status_code=400, detail=res["message"])
        
    return FaceRecognitionResponse(**res)
