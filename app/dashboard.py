import os
import sys
import io
import time
import base64
import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"

st.set_page_config(
    page_title="Smart Retail Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000"
API_KEY = "retail-secret-key-2026"
HEADERS = {"X-API-Key": API_KEY}

@st.cache_resource(show_spinner=False)
def check_api_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/dashboard/stats", timeout=2)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    return False

@st.cache_resource(show_spinner=False)
def load_local_services():
    try:
        from app.services.cv_service import CVService
        from app.services.nlp_service import NLPService
        from app.services.chatbot_service import ChatbotService
        
        return {
            "cv": CVService(),
            "nlp": NLPService(),
            "chatbot": ChatbotService()
        }
    except Exception as e:
        st.error(f"Failed to load local services fallback: {e}")
        return None

is_api_online = check_api_connection()
local_services = None
if not is_api_online:
    local_services = load_local_services()

st.markdown("""
<style>
    /* Dark Slate Background & Glassmorphism Theme */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Elegant Title and Header Gradients */
    .main-title {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0px;
        text-shadow: 0px 4px 10px rgba(0, 242, 254, 0.15);
    }
    .sub-title {
        color: #8b949e;
        font-size: 1.1rem;
        margin-top: 0px;
        margin-bottom: 25px;
    }
    
    /* Custom Neon Metric Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(56, 139, 253, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 139, 253, 0.6);
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #58a6ff;
    }
    
    /* Connection Badges */
    .status-badge-api {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-badge-local {
        background: linear-gradient(135deg, #f12711 0%, #f5af19 100%);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid rgba(48, 54, 61, 0.8) !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1542744094-3a31f103e35f?q=80&w=200", use_container_width=True)
    st.markdown("<h2 style='text-align: center; color: #58a6ff;'>System Controls</h2>", unsafe_allow_html=True)
    
    if is_api_online:
        st.markdown("<div style='text-align: center;'><span class='status-badge-api'>🟢 REST API CONNECTED</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align: center;'><span class='status-badge-local'>🟠 LOCAL FALLBACK ACTIVE</span></div>", unsafe_allow_html=True)
        
    st.markdown("<hr style='border-color: rgba(48,54,61,0.8);'/>", unsafe_allow_html=True)
    
    st.markdown("### Model Versions")
    st.write("🤖 **Face Recognizer**: OpenCV LBPH (v1.0)")
    st.write("📦 **Product Classifier**: PyTorch MobileNetV2")
    st.write("💬 **Sentiment/Chatbot**: TF-IDF + Logistic Reg.")
    
    st.markdown("<hr style='border-color: rgba(48,54,61,0.8);'/>", unsafe_allow_html=True)
    
    st.markdown("### Quick Resources")
    st.markdown("[📖 API Swagger Docs (Local)](http://localhost:8000/docs)")
    st.markdown("[📂 GitHub Repository](https://github.com)")
    
    st.info("Ensure the FastAPI backend is running via `uvicorn app.main:app --reload` to unlock full API gateway features (REST client authentication, WebSocket stream preparation).")

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown("<h1 class='main-title'>Smart Retail Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Unified Customer Loyalty Analytics & NLP Feedback Processing Platform</p>", unsafe_allow_html=True)
with col2:
    if st.button("🔄 Refresh Data"):
        st.rerun()

stats = None
if is_api_online:
    try:
        response = requests.get(f"{API_BASE_URL}/dashboard/stats", headers=HEADERS)
        if response.status_code == 200:
            stats = response.json()
    except Exception as e:
        st.warning(f"Error fetching stats from API: {e}")
        
if stats is None:
    from app.services.cv_service import VISIT_LOGS as v_logs
    from app.services.nlp_service import SENTIMENT_LOGS as s_logs
    from app.services.chatbot_service import CHATBOT_LOGS as c_logs
    
    total_visits = len(v_logs)
    known = len([v for v in v_logs if v["customer_id"] != -1])
    unknown = total_visits - known
    total_s = len(s_logs)
    
    dist = {"positive": 0, "negative": 0, "neutral": 0}
    sum_c = 0.0
    for s in s_logs:
        dist[s["sentiment"]] = dist.get(s["sentiment"], 0) + 1
        sum_c += s["confidence"]
    avg_c = (sum_c / total_s) if total_s > 0 else 0.0
    
    stats = {
        "total_visits": total_visits,
        "known_customer_visits": known,
        "unknown_customer_visits": unknown,
        "total_sentiments": total_s,
        "sentiment_distribution": dist,
        "avg_sentiment_confidence": round(avg_c, 4),
        "visit_logs": list(reversed(v_logs)),
        "sentiment_logs": list(reversed(s_logs)),
        "chatbot_logs": list(reversed(c_logs))
    }

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Customer Visits</div>
        <div class="metric-value">{stats['total_visits']}</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Loyalty Members Recognized</div>
        <div class="metric-value">{stats['known_customer_visits']}</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Sentiment Volume</div>
        <div class="metric-value">{stats['total_sentiments']}</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    sentiment_health = "N/A"
    if stats['total_sentiments'] > 0:
        pos_pct = (stats['sentiment_distribution'].get('positive', 0) / stats['total_sentiments']) * 100
        sentiment_health = f"{pos_pct:.0f}% Pos"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Sentiment Health Score</div>
        <div class="metric-value" style="color: #38ef7d;">{sentiment_health}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

tab_vision, tab_nlp, tab_chatbot, tab_analytics = st.tabs([
    "📸 Computer Vision Suite", 
    "📈 Sentiment Analytics", 
    "💬 Support Chatbot", 
    "📊 Loyalty & Visit Analytics"
])

with tab_vision:
    st.markdown("### Image Processing and Deep Learning Suite")
    st.write("Test OpenCV image preprocessing operations, Face Recognition (LBPH), and Product Category classification (MobileNetV2).")
    
    cv_col1, cv_col2 = st.columns([0.4, 0.6])
    
    with cv_col1:
        st.markdown("#### Input Source")
        img_input = st.radio("Choose Input Type:", ["Upload Image File", "Use Live Camera Mock"])
        
        uploaded_file = None
        img_bytes = None
        
        if img_input == "Upload Image File":
            uploaded_file = st.file_uploader("Upload product or customer face image:", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                img_bytes = uploaded_file.read()
                st.image(Image.open(io.BytesIO(img_bytes)), caption="Original Uploaded Image", use_container_width=True)
        else:
            mock_option = st.selectbox(
                "Select a mock test image:",
                [
                    "Select...",
                    "Customer: Alice (ID 1)",
                    "Customer: Bob (ID 2)",
                    "Customer: Charlie (ID 3)",
                    "Product: Grocery Item (Apple)",
                    "Product: Electronic Device (Phone)",
                    "Product: Leather Bag"
                ]
            )
            
            if mock_option != "Select...":
                if "Customer:" in mock_option:
                    cid = 1 if "Alice" in mock_option else (2 if "Bob" in mock_option else 3)
                    from app.models.train_face_recognizer import generate_synthetic_face
                    img_np = generate_synthetic_face(cid, 5)
                    import cv2
                    _, buffer = cv2.imencode(".png", img_np)
                    img_bytes = buffer.tobytes()
                    st.image(Image.open(io.BytesIO(img_bytes)), caption=f"Mock Face: {mock_option}", use_container_width=True)
                else:
                    class_idx = 4 if "Grocery" in mock_option else (2 if "Electronic" in mock_option else 1)
                    from app.models.train_product_classifier import generate_synthetic_product
                    tensor = generate_synthetic_product(class_idx, 5)
                    img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    import cv2
                    _, buffer = cv2.imencode(".png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                    img_bytes = buffer.tobytes()
                    st.image(Image.open(io.BytesIO(img_bytes)), caption=f"Mock Product: {mock_option}", use_container_width=True)
        
    with cv_col2:
        if img_bytes is not None:
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                run_preprocess = st.button("⚙️ Apply OpenCV Pipeline")
            with action_col2:
                run_face = st.button("👤 Recognize Customer Face")
            with action_col3:
                run_product = st.button("🏷️ Classify Product Category")
                
            if run_preprocess:
                st.markdown("#### OpenCV Basics Preprocessing Steps")
                with st.spinner("Processing image through OpenCV..."):
                    try:
                        if is_api_online:
                            files = {"file": ("image.png", img_bytes, "image/png")}
                            response = requests.post(f"{API_BASE_URL}/vision/preprocess-image", files=files, headers=HEADERS)
                            res = response.json()
                            gray_img = Image.open(io.BytesIO(base64.b64decode(res["gray_image_base64"])))
                            edges_img = Image.open(io.BytesIO(base64.b64decode(res["edges_image_base64"])))
                            bbox_img = Image.open(io.BytesIO(base64.b64decode(res["bbox_image_base64"])))
                            num_faces = res["num_faces"]
                        else:
                            res = local_services["cv"].apply_cv_ops(img_bytes)
                            gray_img = Image.open(io.BytesIO(res["gray_bytes"]))
                            edges_img = Image.open(io.BytesIO(res["edges_bytes"]))
                            bbox_img = Image.open(io.BytesIO(res["bbox_bytes"]))
                            num_faces = res["num_faces"]
                            
                        p_col1, p_col2 = st.columns(2)
                        with p_col1:
                            st.image(gray_img, caption="1. Grayscale & Resized", use_container_width=True)
                            st.image(bbox_img, caption="3. Bounding Boxes (Haar Cascade)", use_container_width=True)
                        with p_col2:
                            st.image(edges_img, caption="2. Blurred & Canny Edges", use_container_width=True)
                            st.success(f"Faces Detected by Haar Cascade: {num_faces}")
                    except Exception as e:
                        st.error(f"Error during OpenCV ops: {e}")
                        
            elif run_face:
                st.markdown("#### Face Recognition Result (LBPH)")
                with st.spinner("Analyzing face encodings..."):
                    try:
                        if is_api_online:
                            files = {"file": ("image.png", img_bytes, "image/png")}
                            response = requests.post(f"{API_BASE_URL}/vision/recognize-face", files=files, headers=HEADERS)
                            res = response.json()
                        else:
                            res = local_services["cv"].recognize_customer(img_bytes)
                            
                        if res.get("recognized", False):
                            st.balloons()
                            st.markdown(f"""
                            <div style="background: rgba(56, 139, 253, 0.1); border: 2px solid #58a6ff; border-radius: 12px; padding: 25px; margin-top: 15px;">
                                <h3 style="color: #58a6ff; margin-top: 0px;">Welcome Back, {res['name']}!</h3>
                                <p>👤 <b>Customer ID</b>: {res['customer_id']}</p>
                                <p>⭐ <b>Loyalty Club Points</b>: <span style="font-size: 1.2rem; color: #38ef7d; font-weight: 700;">{res['loyalty_points']}</span></p>
                                <p>⏰ <b>Last Visited</b>: {res['last_visit']}</p>
                                <p>📉 <b>LBPH Distance Confidence</b>: {res['confidence_score']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background: rgba(248, 81, 73, 0.1); border: 2px solid #f85149; border-radius: 12px; padding: 25px; margin-top: 15px; margin-bottom: 15px;">
                                <h3 style="color: #f85149; margin-top: 0px;">Unknown Visitor Detected</h3>
                                <p>We could not match this face to any consenting member in our loyalty database.</p>
                                <p>📉 <b>Confidence score</b>: {res.get('confidence_score', 0.0):.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.session_state.show_register = True
                            st.session_state.temp_img_bytes = img_bytes
                    except Exception as e:
                        st.error(f"Error during face recognition: {e}")
                        
            elif run_product:
                st.markdown("#### Product Classification Result (MobileNetV2)")
                with st.spinner("Running deep learning forward pass..."):
                    try:
                        if is_api_online:
                            files = {"file": ("image.png", img_bytes, "image/png")}
                            response = requests.post(f"{API_BASE_URL}/vision/classify-product", files=files, headers=HEADERS)
                            res = response.json()
                        else:
                            res = local_services["cv"].classify_product(img_bytes)
                            
                        st.success(f"Predicted Category: **{res['predicted_category'].upper()}** (Confidence: {res['confidence_score'] * 100:.2f}%)")
                        
                        probs = res["category_probabilities"]
                        df_probs = pd.DataFrame(list(probs.items()), columns=["Category", "Probability"])
                        st.bar_chart(df_probs.set_index("Category"))
                    except Exception as e:
                        st.error(f"Error during product classification: {e}")
            
            if st.session_state.get("show_register", False) and st.session_state.get("temp_img_bytes") is not None:
                st.write("---")
                st.markdown("### 📝 Register New Loyalty Profile")
                st.write("This visitor is not in our database. Enter their name to register them and retrain the face recognition system on-the-fly.")
                
                with st.form("register_form"):
                    new_cust_name = st.text_input("Full Name", placeholder="e.g. Dr. Jane Smith")
                    submit_reg = st.form_submit_button("Submit Registration & Retrain")
                    
                    if submit_reg:
                        if not new_cust_name.strip():
                            st.error("Please enter a valid name.")
                        else:
                            with st.spinner("Uploading biometric profile & retraining models..."):
                                try:
                                    if is_api_online:
                                        files = {"file": ("image.png", st.session_state.temp_img_bytes, "image/png")}
                                        data = {"name": new_cust_name}
                                        response = requests.post(f"{API_BASE_URL}/vision/register-face", files=files, data=data, headers=HEADERS)
                                        if response.status_code == 200:
                                            reg_res = response.json()
                                            st.success(f"Success! Registered {new_cust_name} with Loyalty ID: {reg_res['customer_id']}. Please run face recognition again to test!")
                                            st.balloons()
                                            st.session_state.show_register = False
                                            st.session_state.temp_img_bytes = None
                                        else:
                                            st.error(f"Registration failed: {response.json().get('detail', 'Unknown error')}")
                                    else:
                                        reg_res = local_services["cv"].register_customer(st.session_state.temp_img_bytes, new_cust_name)
                                        if "status" in reg_res and reg_res["status"] == "error":
                                            st.error(reg_res["message"])
                                        else:
                                            st.success(reg_res["message"])
                                            st.balloons()
                                            st.session_state.show_register = False
                                            st.session_state.temp_img_bytes = None
                                except Exception as e:
                                    st.error(f"Error during registration: {e}")
        else:
            st.info("👈 Please upload an image or choose a live camera mock sample in the left column to run predictions.")

with tab_nlp:
    st.markdown("### Customer Review Sentiment Classifier")
    st.write("Type a customer review or purchase feedback to predict the sentiment (Positive, Negative, Neutral) using the TF-IDF + Logistic Regression pipeline.")
    
    nlp_col1, nlp_col2 = st.columns([0.5, 0.5])
    
    with nlp_col1:
        st.markdown("#### Enter Feedback")
        review_text = st.text_area(
            "Type customer feedback:",
            placeholder="Type something like: 'The jacket is warm and fits beautifully, shipping was very fast!'",
            height=120
        )
        run_sentiment = st.button("🔍 Analyze Feedback Sentiment")
        
    with nlp_col2:
        if run_sentiment and review_text.strip():
            with st.spinner("Analyzing text and cleaning tokens..."):
                try:
                    if is_api_online:
                        payload = {"text": review_text}
                        response = requests.post(f"{API_BASE_URL}/nlp/analyze-sentiment", json=payload, headers=HEADERS)
                        res = response.json()
                    else:
                        res = local_services["nlp"].analyze_sentiment(review_text)
                        
                    sentiment = res["sentiment"]
                    confidence = res["confidence"]
                    probs = res["probabilities"]
                    
                    sentiment_color = "#38ef7d" if sentiment == "positive" else ("#f85149" if sentiment == "negative" else "#f5af19")
                    
                    st.markdown(f"""
                    <div style="background: rgba(22, 27, 34, 0.8); border: 2px solid {sentiment_color}; border-radius: 12px; padding: 20px; text-align: center;">
                        <h4 style="color: #8b949e; text-transform: uppercase; margin: 0;">Predicted Sentiment</h4>
                        <h2 style="color: {sentiment_color}; margin: 10px 0; font-size: 2.2rem; font-weight: 800;">{sentiment.upper()}</h2>
                        <p style="margin: 0; color: #8b949e;">Model Confidence: <b>{confidence * 100:.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    df_s_probs = pd.DataFrame(list(probs.items()), columns=["Sentiment Class", "Probability"])
                    st.bar_chart(df_s_probs.set_index("Sentiment Class"))
                except Exception as e:
                    st.error(f"Error during sentiment analysis: {e}")
        else:
            st.info("Write some feedback text in the text box and press analyze to test model classification.")

with tab_chatbot:
    st.markdown("### FAQ Hybrid Chatbot Widget")
    st.write("Chat with our smart support bot. It resolves shipping tracking, shop times, and return requests using exact rule matching with ML intent fallback.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Smart Retail assistant. Ask me about store hours, return policies, order tracking, or payment methods!"}
        ]
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    user_query = st.chat_input("Ask a question (e.g. 'when do you close today?' or 'track my order')")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.spinner("Thinking..."):
            try:
                if is_api_online:
                    payload = {"message": user_query}
                    response = requests.post(f"{API_BASE_URL}/chatbot/", json=payload, headers=HEADERS)
                    res = response.json()
                else:
                    res = local_services["chatbot"].get_reply(user_query)
                    
                bot_reply = res["reply"]
                intent = res["intent"]
                strategy = res["strategy"]
                conf = res["confidence"]
                
                with st.chat_message("assistant"):
                    st.write(bot_reply)
                    st.caption(f"Intent classified: **{intent}** | Confidence: **{conf:.2f}** | Strategy: **{strategy.upper()}**")
                
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                st.error(f"Error contacting chatbot service: {e}")

with tab_analytics:
    st.markdown("### Store Loyalty Analytics Dashboard")
    st.write("Aggregated visual reporting of customer demographics, loyalty visits, and sentiment health drift.")
    
    an_col1, an_col2 = st.columns(2)
    
    with an_col1:
        st.markdown("#### Sentiment Distribution")
        s_dist = stats["sentiment_distribution"]
        df_dist = pd.DataFrame(list(s_dist.items()), columns=["Sentiment", "Count"])
        st.bar_chart(df_dist.set_index("Sentiment"))
        
        st.markdown("#### Customer Loyalty Visit Logs")
        df_visits = pd.DataFrame(stats["visit_logs"])
        st.dataframe(df_visits, use_container_width=True)
        
    with an_col2:
        st.markdown("#### Customer Visits Breakdown")
        df_breakdown = pd.DataFrame({
            "Customer Segment": ["Recognized Loyalty Members", "Unknown Guest Visitors"],
            "Count": [stats["known_customer_visits"], stats["unknown_customer_visits"]]
        })
        st.bar_chart(df_breakdown.set_index("Customer Segment"))
        
        st.markdown("#### Customer Sentiment Logs (Latest Feed)")
        df_sent = pd.DataFrame(stats["sentiment_logs"])
        if not df_sent.empty:
            st.dataframe(df_sent[["timestamp", "text", "sentiment", "confidence"]], use_container_width=True)
        else:
            st.info("No text sentiment feedback recorded yet.")
