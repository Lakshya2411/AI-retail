import os
import re
import csv
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK data
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Preprocessing helpers
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation & numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(cleaned_words)

def train_sentiment():
    print("Loading data/reviews.csv...")
    reviews = []
    labels = []
    
    with open("data/reviews.csv", mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if len(row) >= 2:
                reviews.append(row[0])
                labels.append(row[1])
                
    print(f"Loaded {len(reviews)} reviews. Preprocessing...")
    preprocessed_reviews = [preprocess_text(r) for r in reviews]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_reviews, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # TF-IDF Vectorization
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train_vectorized, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vectorized)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create models directory if it doesn't exist
    os.makedirs("app/models", exist_ok=True)
    
    # Save model and vectorizer
    print("Saving model and vectorizer to app/models/...")
    joblib.dump(model, "app/models/sentiment_model.pkl")
    joblib.dump(vectorizer, "app/models/vectorizer.pkl")
    print("Sentiment model training complete!")

if __name__ == "__main__":
    train_sentiment()
