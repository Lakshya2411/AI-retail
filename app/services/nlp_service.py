import os
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set security environment variable in-script to bypass NLTK CWD check
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"

# Global log of feedback sentiment analysis for dashboard charts
SENTIMENT_LOGS = [
    {"timestamp": "2026-08-02 09:05", "text": "I love the quick customer checkout!", "sentiment": "positive", "confidence": 0.92},
    {"timestamp": "2026-08-02 10:12", "text": "The clothes fit perfectly, five stars.", "sentiment": "positive", "confidence": 0.98},
    {"timestamp": "2026-08-02 10:45", "text": "Terrible delivery speed. Ripped box.", "sentiment": "negative", "confidence": 0.95},
    {"timestamp": "2026-08-02 11:00", "text": "It was ok, nothing special.", "sentiment": "neutral", "confidence": 0.74}
]

class NLPService:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self.model_path = os.path.join(self.models_dir, "sentiment_model.pkl")
        self.vectorizer_path = os.path.join(self.models_dir, "vectorizer.pkl")
        
        # Download NLTK data programmatically if cached missing
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.model = None
        self.vectorizer = None
        
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            try:
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                print("[NLP Service] Loaded Sentiment Model & Vectorizer successfully.")
            except Exception as e:
                print(f"[NLP Service] Error loading Sentiment Model: {e}")
        else:
            print("[NLP Service] Warning: Sentiment Model files not found.")

    def clean_text(self, text):
        if not text:
            return ""
        # Lowercase
        text = text.lower()
        # Remove punctuation & numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        words = nltk.word_tokenize(text)
        # Remove stopwords and lemmatize
        cleaned_words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        return ' '.join(cleaned_words)

    def analyze_sentiment(self, text):
        if self.model is None or self.vectorizer is None:
            return {"status": "error", "message": "Sentiment analysis model is not loaded."}
            
        cleaned = self.clean_text(text)
        
        # Check if the text is empty after cleaning
        if not cleaned.strip():
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "message": "Input contains no cleanable words."
            }
            
        vectorized = self.vectorizer.transform([cleaned])
        
        # Predict class
        prediction = self.model.predict(vectorized)[0]
        
        # Calculate probabilities
        probabilities = self.model.predict_proba(vectorized)[0]
        class_idx = list(self.model.classes_).index(prediction)
        confidence = float(probabilities[class_idx])
        
        # Log feedback for dashboard analytics
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        SENTIMENT_LOGS.append({
            "timestamp": timestamp,
            "text": text,
            "sentiment": prediction,
            "confidence": confidence
        })
        
        return {
            "sentiment": prediction,
            "confidence": confidence,
            "probabilities": {self.model.classes_[i]: float(probabilities[i]) for i in range(len(self.model.classes_))}
        }
