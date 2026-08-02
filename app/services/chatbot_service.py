import os
import re
import json
import random
import joblib
import nltk
from nltk.stem import WordNetLemmatizer

os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"

CHATBOT_LOGS = [
    {"timestamp": "2026-08-02 09:10", "query": "hello", "response": "Hi there! How can I help you today?", "intent": "greeting"},
    {"timestamp": "2026-08-02 10:15", "query": "when does the shop open", "response": "Our stores are open Monday through Saturday from 9:00 AM to 9:00 PM, and Sundays from 10:00 AM to 6:00 PM.", "intent": "store_hours"}
]

class ChatbotService:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        
        self.intents_path = os.path.join(self.data_dir, "intents.json")
        self.model_path = os.path.join(self.models_dir, "chatbot_model.pkl")
        self.vectorizer_path = os.path.join(self.models_dir, "chatbot_vectorizer.pkl")
        
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        
        self.intents_data = {}
        if os.path.exists(self.intents_path):
            with open(self.intents_path, "r", encoding="utf-8") as f:
                self.intents_data = json.load(f)
        else:
            print("[Chatbot Service] Warning: intents.json file not found.")
            
        self.model = None
        self.vectorizer = None
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            try:
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                print("[Chatbot Service] Loaded Chatbot Classification Model & Vectorizer successfully.")
            except Exception as e:
                print(f"[Chatbot Service] Error loading Chatbot Model: {e}")
        else:
            print("[Chatbot Service] Warning: Chatbot classifier model files not found.")

    def clean_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = nltk.word_tokenize(text)
        cleaned_words = [self.lemmatizer.lemmatize(w) for w in words]
        return ' '.join(cleaned_words)

    def get_intent_response(self, tag):
        for intent in self.intents_data.get("intents", []):
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I'm sorry, I don't have a configured response for that category."

    def get_reply(self, message):
        """
        Processes message and returns a reply.
        First checks rules, then falls back to the ML classification model.
        """
        if not message or not message.strip():
            return {
                "reply": "Please type a message to start chatting!",
                "intent": "empty",
                "confidence": 1.0,
                "strategy": "rule"
            }
            
        cleaned = self.clean_text(message)
        
        greetings_keywords = ["hi", "hello", "hey", "greetings", "howdy", "good morning", "good day"]
        words_set = set(cleaned.split())
        if any(greet in words_set for greet in greetings_keywords):
            reply = self.get_intent_response("greeting")
            self._log_chat(message, reply, "greeting")
            return {
                "reply": reply,
                "intent": "greeting",
                "confidence": 1.0,
                "strategy": "rule"
            }
            
        goodbye_keywords = ["bye", "goodbye", "exit", "quit", "see you"]
        if any(bye in words_set for bye in goodbye_keywords):
            reply = self.get_intent_response("goodbye")
            self._log_chat(message, reply, "goodbye")
            return {
                "reply": reply,
                "intent": "goodbye",
                "confidence": 1.0,
                "strategy": "rule"
            }
            
        order_keywords = ["track", "package", "shipment", "delivery", "order"]
        if any(word in words_set for word in order_keywords):
            reply = self.get_intent_response("order_status")
            self._log_chat(message, reply, "order_status")
            return {
                "reply": reply,
                "intent": "order_status",
                "confidence": 1.0,
                "strategy": "rule"
            }

        return_keywords = ["return", "refund", "exchange", "returned", "policy"]
        if any(word in words_set for word in return_keywords):
            reply = self.get_intent_response("return_policy")
            self._log_chat(message, reply, "return_policy")
            return {
                "reply": reply,
                "intent": "return_policy",
                "confidence": 1.0,
                "strategy": "rule"
            }

        hours_keywords = ["hour", "hours", "open", "close", "timing", "timings", "weekend", "weekends"]
        if any(word in words_set for word in hours_keywords):
            reply = self.get_intent_response("store_hours")
            self._log_chat(message, reply, "store_hours")
            return {
                "reply": reply,
                "intent": "store_hours",
                "confidence": 1.0,
                "strategy": "rule"
            }

        payment_keywords = ["payment", "pay", "card", "cash", "paypal", "apple", "google"]
        if any(word in words_set for word in payment_keywords):
            reply = self.get_intent_response("payment_methods")
            self._log_chat(message, reply, "payment_methods")
            return {
                "reply": reply,
                "intent": "payment_methods",
                "confidence": 1.0,
                "strategy": "rule"
            }

        contact_keywords = ["contact", "support", "call", "phone", "email", "help", "agent", "human", "person"]
        if any(word in words_set for word in contact_keywords):
            reply = self.get_intent_response("contact_support")
            self._log_chat(message, reply, "contact_support")
            return {
                "reply": reply,
                "intent": "contact_support",
                "confidence": 1.0,
                "strategy": "rule"
            }
            
        if self.model is None or self.vectorizer is None:
            reply = "Our automated bot is offline, but you can email us at support@smartretail.com for help!"
            self._log_chat(message, reply, "fallback")
            return {
                "reply": reply,
                "intent": "unknown",
                "confidence": 0.0,
                "strategy": "fallback"
            }
            
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        
        probabilities = self.model.predict_proba(vectorized)[0]
        class_idx = list(self.model.classes_).index(prediction)
        confidence = float(probabilities[class_idx])
        
        confidence_threshold = 0.40
        
        if confidence >= confidence_threshold:
            reply = self.get_intent_response(prediction)
            intent_matched = prediction
            strategy = "ml"
        else:
            reply = "I'm sorry, I'm not sure I understand. For help with orders, returns, or shop hours, please ask, or contact support at support@smartretail.com."
            intent_matched = "unknown_fallback"
            strategy = "fallback"
            
        self._log_chat(message, reply, intent_matched)
        
        return {
            "reply": reply,
            "intent": intent_matched,
            "confidence": confidence,
            "strategy": strategy
        }

    def _log_chat(self, query, response, intent):
        import time
        CHATBOT_LOGS.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "query": query,
            "response": response,
            "intent": intent
        })
