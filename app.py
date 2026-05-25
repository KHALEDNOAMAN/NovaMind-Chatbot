"""
NovaMind AI Assistant — Powered by Groq (Llama 3)
==================================================
An intelligent conversational AI assistant that combines local NLP 
analysis with Groq's ultra-fast Llama 3 language model for smart,
context-aware responses.

Features:
  - Groq API (Llama 3) for intelligent responses
  - Local NLP: intent classification, sentiment analysis, entity recognition
  - Conversation memory & context awareness
  - Beautiful real-time web interface with NLP analytics
  - Fallback to local knowledge base when API is unavailable

Author: Khaled Noaman
Technologies: Python, Flask, Groq API, NLTK, Scikit-learn
"""

import json
import os
import random
import re
import string
from datetime import datetime

import numpy as np
import requests as http_requests
from flask import Flask, render_template, request, jsonify, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ============================================
# AI API Configuration (Groq - Free & Fast)
# ============================================
def load_api_key():
    """Load API key from .env file or environment variable."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() == 'GROQ_API_KEY':
                        return value.strip()
    return os.environ.get("GROQ_API_KEY", "")

GROQ_API_KEY = load_api_key()
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# System prompt
SYSTEM_PROMPT = """You are NovaMind, a friendly and intelligent AI assistant created by Khaled Noaman, 
a Computer Engineering student at Istanbul Arel University in Istanbul, Turkey.

Your personality:
- You are helpful, knowledgeable, and conversational
- You use emojis occasionally to be friendly (but not excessively)
- You give concise but informative answers (2-4 sentences for simple questions, more for complex ones)
- You can discuss any topic: science, technology, programming, math, general knowledge, etc.
- When asked about yourself, mention you were built with Python, Flask, NLTK, and Groq (Llama 3)
- When asked about your creator, mention Khaled Noaman and his GitHub: github.com/KhaledNoaman

Keep responses conversational and engaging. Don't use markdown headers or bullet points unless specifically helpful."""


class GroqClient:
    """Client for Groq API (OpenAI-compatible, uses Llama 3)."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.conversation_histories = {}
    
    def is_available(self):
        return bool(self.api_key) and len(self.api_key) > 10
    
    def get_response(self, user_message, session_id="default"):
        if not self.is_available():
            return None
        
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = []
        
        history = self.conversation_histories[session_id]
        
        # Build messages array (OpenAI format)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history (last 10 exchanges)
        for exchange in history[-10:]:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["bot"]})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 500,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = http_requests.post(GROQ_URL, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data["choices"][0]["message"]["content"]
                history.append({"user": user_message, "bot": bot_response})
                return bot_response
            else:
                print(f"[Groq API Error] Status {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"[Groq API Error] {e}")
            return None


# ============================================
# Local Knowledge Base (Fallback)
# ============================================
KNOWLEDGE_BASE = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "what's up", "howdy", "greetings", "salam", "merhaba"],
        "responses": [
            "Hello! I'm NovaMind, your AI assistant. How can I help you today? 🤖",
            "Hey there! Nice to meet you. What can I do for you? 😊",
            "Hi! I'm ready to assist you. What's on your mind? 💡",
        ]
    },
    "farewell": {
        "patterns": ["bye", "goodbye", "see you", "later", "take care", "farewell", "good night", "ciao"],
        "responses": [
            "Goodbye! It was great talking to you. Come back anytime! 👋",
            "See you later! Have a wonderful day! 🌟",
            "Take care! I'll be here whenever you need me. 😊",
        ]
    },
    "identity": {
        "patterns": ["who are you", "what are you", "what's your name", "tell me about yourself", "introduce yourself", "your name", "are you a bot", "are you ai"],
        "responses": [
            "I'm NovaMind, an AI assistant created by Khaled Noaman! I'm powered by Groq (Llama 3) and local NLP. I can chat, analyze sentiment, detect entities, and answer questions on any topic! 🧠",
        ]
    },
    "jokes": {
        "patterns": ["tell me a joke", "joke", "make me laugh", "funny", "humor"],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛😄",
            "Why was the JavaScript developer sad? Because he didn't Node how to Express himself! 😂",
            "What's a computer's favorite snack? Microchips! 🍟💻",
            "How many programmers does it take to change a light bulb? None — that's a hardware problem! 💡😂",
        ]
    },
    "thanks": {
        "patterns": ["thank you", "thanks", "thx", "appreciate it", "thanks a lot"],
        "responses": [
            "You're welcome! Happy to help! 😊",
            "Anytime! That's what I'm here for! 🌟",
        ]
    },
    "creator": {
        "patterns": ["who made you", "who created you", "who built you", "your creator", "who programmed you"],
        "responses": [
            "I was created by Khaled Noaman, a Computer Engineering student at Istanbul Arel University! Built with Python, Flask, Groq (Llama 3), NLTK, and Scikit-learn. Check out his work at github.com/KhaledNoaman 🚀",
        ]
    }
}


# ============================================
# NLP Engine (Local Analysis)
# ============================================
class NLPEngine:
    """Local NLP engine for intent classification, sentiment, and entity extraction."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()
        self.vectorizer = TfidfVectorizer()
        self._build_intent_model()
    
    def _build_intent_model(self):
        self.intent_labels = []
        self.intent_patterns = []
        for intent, data in KNOWLEDGE_BASE.items():
            for pattern in data["patterns"]:
                self.intent_labels.append(intent)
                self.intent_patterns.append(pattern.lower())
        self.tfidf_matrix = self.vectorizer.fit_transform(self.intent_patterns)
    
    def preprocess(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 1]
        return ' '.join(tokens)
    
    def classify_intent(self, text):
        processed = self.preprocess(text)
        if not processed:
            processed = text.lower()
        user_vector = self.vectorizer.transform([processed])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        if best_score > 0.2:
            return self.intent_labels[best_idx], best_score
        return None, 0.0
    
    def analyze_sentiment(self, text):
        positive_words = {'good', 'great', 'awesome', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy', 'joy', 'beautiful', 'brilliant', 'perfect', 'best', 'nice', 'cool', 'super', 'like', 'enjoy', 'thank', 'thanks', 'helpful', 'impressive'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'sad', 'angry', 'worst', 'ugly', 'stupid', 'boring', 'annoying', 'disappointed', 'frustrating', 'difficult', 'problem', 'wrong', 'fail', 'error', 'broken', 'useless', 'poor'}
        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        if pos_count > neg_count:
            return "positive", min(pos_count * 0.3, 1.0)
        elif neg_count > pos_count:
            return "negative", min(neg_count * 0.3, 1.0)
        return "neutral", 0.0
    
    def extract_entities(self, text):
        entities = []
        for n in re.findall(r'\b\d+\.?\d*\b', text):
            entities.append({"type": "NUMBER", "value": n})
        for e in re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text):
            entities.append({"type": "EMAIL", "value": e})
        for u in re.findall(r'https?://\S+', text):
            entities.append({"type": "URL", "value": u})
        for d in re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text):
            entities.append({"type": "DATE", "value": d})
        common = {'I','The','This','That','What','How','Why','When','Where','Who','Can','Could','Would','Should','Hello','Hi','Hey','Please','Thanks','Thank','Yes','No','Ok'}
        for c in re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text):
            if c not in common:
                entities.append({"type": "PROPER_NOUN", "value": c})
        return entities


# ============================================
# Conversation Manager
# ============================================
class ConversationManager:
    def __init__(self):
        self.nlp = NLPEngine()
        self.ai = GroqClient(GROQ_API_KEY)
        self.conversations = {}
        
        mode = "Groq AI (Llama 3)" if self.ai.is_available() else "Local Knowledge Base"
        print(f"  Response Mode: {mode}")
    
    def get_response(self, user_message, session_id="default"):
        if session_id not in self.conversations:
            self.conversations[session_id] = {"history": [], "message_count": 0, "start_time": datetime.now().isoformat()}
        
        context = self.conversations[session_id]
        context["message_count"] += 1
        
        # Local NLP analysis (always runs)
        intent, confidence = self.nlp.classify_intent(user_message)
        sentiment, sent_score = self.nlp.analyze_sentiment(user_message)
        entities = self.nlp.extract_entities(user_message)
        
        # Get response — try Groq AI first, fall back to local
        response = None
        source = "local"
        
        if self.ai.is_available():
            response = self.ai.get_response(user_message, session_id)
            if response:
                source = "groq"
        
        # Fallback to local knowledge base
        if not response:
            if intent and intent in KNOWLEDGE_BASE:
                response = random.choice(KNOWLEDGE_BASE[intent]["responses"])
                if intent == "time":
                    response = f"The current time is {datetime.now().strftime('%I:%M %p')} ⏰"
                elif intent == "date":
                    response = f"Today is {datetime.now().strftime('%A, %B %d, %Y')} 📅"
            else:
                response = "I'm having trouble connecting to my AI brain right now. Please make sure the Groq API key is set in the .env file. Get a free key at console.groq.com/keys 🔑"
        
        # Store in history
        context["history"].append({
            "user": user_message, "bot": response, "intent": intent,
            "confidence": float(confidence) if confidence else 0,
            "sentiment": sentiment, "entities": entities,
            "source": source, "timestamp": datetime.now().isoformat()
        })
        
        return {
            "response": response,
            "intent": intent or "general",
            "confidence": round(float(confidence), 3) if confidence else 0,
            "sentiment": {"label": sentiment, "score": round(sent_score, 3)},
            "entities": entities,
            "source": source,
            "message_count": context["message_count"]
        }
    
    def get_stats(self, session_id="default"):
        if session_id not in self.conversations:
            return {"message_count": 0}
        ctx = self.conversations[session_id]
        sentiments = [h["sentiment"] for h in ctx["history"]]
        return {
            "message_count": ctx["message_count"],
            "sentiment_summary": {"positive": sentiments.count("positive"), "negative": sentiments.count("negative"), "neutral": sentiments.count("neutral")},
            "total_entities": sum(len(h.get("entities", [])) for h in ctx["history"]),
            "ai_active": self.ai.is_available()
        }


# ============================================
# Flask Routes
# ============================================
chatbot = ConversationManager()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    session_id = session.get('session_id', 'default')
    if 'session_id' not in session:
        session['session_id'] = os.urandom(8).hex()
        session_id = session['session_id']
    result = chatbot.get_response(user_message, session_id)
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def stats():
    session_id = session.get('session_id', 'default')
    return jsonify(chatbot.get_stats(session_id))

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    sentiment, score = chatbot.nlp.analyze_sentiment(text)
    entities = chatbot.nlp.extract_entities(text)
    intent, confidence = chatbot.nlp.classify_intent(text)
    return jsonify({
        "sentiment": {"label": sentiment, "score": round(score, 3)},
        "entities": entities,
        "intent": {"label": intent or "unknown", "confidence": round(float(confidence), 3)}
    })

if __name__ == '__main__':
    print("=" * 50)
    print("  NovaMind AI Assistant")
    print("  Powered by Groq (Llama 3)")
    print("=" * 50)
    print()
    if chatbot.ai.is_available():
        print("  [OK] Groq AI connected!")
    else:
        print("  [!] No Groq API key found.")
        print("  Add GROQ_API_KEY=your_key to .env file")
    print()
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
