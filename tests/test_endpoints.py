import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

API_KEY = "retail-secret-key-2026"
HEADERS = {"X-API-Key": API_KEY}

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def get_mock_image():
    img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_root_redirect(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307

def test_api_key_unauthorized(client):
    response = client.post("/nlp/analyze-sentiment", json={"text": "Excellent service!"})
    assert response.status_code == 403
    assert "Unauthorized" in response.json()["detail"]

def test_sentiment_endpoint(client):
    response = client.post(
        "/nlp/analyze-sentiment", 
        json={"text": "This product is absolutely amazing! highly recommend"}, 
        headers=HEADERS
    )
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["sentiment"] == "positive"
    assert "probabilities" in res_data
    assert res_data["confidence"] > 0.5

def test_chatbot_endpoint(client):
    response = client.post(
        "/chatbot/",
        json={"message": "What are your operating hours?"},
        headers=HEADERS
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "reply" in res_data
    assert res_data["intent"] in ["store_hours", "unknown", "unknown_fallback"]
    assert "strategy" in res_data

def test_dashboard_stats_endpoint(client):
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    res_data = response.json()
    assert "total_visits" in res_data
    assert "visit_logs" in res_data
    assert "sentiment_logs" in res_data

def test_vision_preprocess(client):
    img_bytes = get_mock_image()
    files = {"file": ("test.png", img_bytes, "image/png")}
    response = client.post("/vision/preprocess-image", files=files, headers=HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert "num_faces" in res_data
    assert "gray_image_base64" in res_data
    assert "edges_image_base64" in res_data

def test_vision_recognize_face(client):
    img_bytes = get_mock_image()
    files = {"file": ("test.png", img_bytes, "image/png")}
    response = client.post("/vision/recognize-face", files=files, headers=HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert "customer_id" in res_data
    assert res_data["customer_id"] == -1
    assert res_data["recognized"] is False

def test_vision_classify_product(client):
    img_bytes = get_mock_image()
    files = {"file": ("test.png", img_bytes, "image/png")}
    response = client.post("/vision/classify-product", files=files, headers=HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert "predicted_category" in res_data
    assert res_data["predicted_category"] in ["shoes", "bags", "electronics", "clothing", "groceries"]
    assert "category_probabilities" in res_data

def test_vision_register_face_no_face(client):
    img_bytes = get_mock_image()
    files = {"file": ("test.png", img_bytes, "image/png")}
    data = {"name": "Test Customer"}
    response = client.post("/vision/register-face", files=files, data=data, headers=HEADERS)
    assert response.status_code == 400
    assert "No face detected" in response.json()["detail"]
