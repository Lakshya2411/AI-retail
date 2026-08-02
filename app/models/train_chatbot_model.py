import os
import re
import json
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set security environment variable in-script to bypass NLTK CWD check
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"

# Download NLTK data (should already be cached)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = nltk.word_tokenize(text)
    cleaned_words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(cleaned_words)

def train_chatbot():
    print("Loading data/intents.json...")
    with open("data/intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    patterns = []
    tags = []
    
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            patterns.append(preprocess_text(pattern))
            tags.append(tag)
            
    print(f"Loaded {len(patterns)} patterns for chatbot intent classification.")
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(patterns)
    y = tags
    
    # Train Logistic Regression
    print("Training intent classification model...")
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X, y)
    
    # Check training accuracy
    train_acc = model.score(X, y)
    print(f"Training accuracy: {train_acc * 100:.2f}%")
    
    # Save model and vectorizer
    os.makedirs("app/models", exist_ok=True)
    print("Saving chatbot models to app/models/...")
    joblib.dump(model, "app/models/chatbot_model.pkl")
    joblib.dump(vectorizer, "app/models/chatbot_vectorizer.pkl")
    print("Chatbot model training complete!")

if __name__ == "__main__":
    train_chatbot()
