"""
AI Assistant Chatbot — NovaMind
================================
An intelligent conversational AI assistant with NLP capabilities,
sentiment analysis, entity recognition, and a beautiful web interface.

Features:
  - Natural Language Processing with intent classification
  - Sentiment analysis on user messages
  - Named Entity Recognition (NER)
  - Context-aware conversation memory
  - Knowledge base with expandable topics
  - Beautiful real-time web interface

Author: Khaled Noaman
Technologies: Python, Flask, NLTK, Scikit-learn, HTML/CSS/JS
"""

import json
import os
import random
import re
import string
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, jsonify, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
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
# Knowledge Base
# ============================================
KNOWLEDGE_BASE = {
    "greetings": {
        "patterns": [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "what's up", "howdy", "greetings", "sup",
            "yo", "hola", "salam", "merhaba"
        ],
        "responses": [
            "Hello! I'm NovaMind, your AI assistant. How can I help you today? 🤖",
            "Hey there! Nice to meet you. What can I do for you? 😊",
            "Hi! I'm ready to assist you. What's on your mind? 💡",
            "Greetings! I'm NovaMind — ask me anything! 🌟"
        ]
    },
    "farewell": {
        "patterns": [
            "bye", "goodbye", "see you", "later", "take care",
            "farewell", "good night", "see ya", "peace out", "ciao"
        ],
        "responses": [
            "Goodbye! It was great talking to you. Come back anytime! 👋",
            "See you later! Have a wonderful day! 🌟",
            "Take care! I'll be here whenever you need me. 😊",
            "Bye! Remember, I'm always here to help! 💫"
        ]
    },
    "identity": {
        "patterns": [
            "who are you", "what are you", "what's your name",
            "tell me about yourself", "introduce yourself",
            "what can you do", "your name", "are you a bot",
            "are you real", "are you ai", "are you human"
        ],
        "responses": [
            "I'm NovaMind, an AI assistant created by Khaled Noaman! I can chat with you, answer questions, analyze your text, and help with various tasks. 🤖",
            "I'm NovaMind — a Python-powered AI chatbot. I use NLP techniques like TF-IDF, sentiment analysis, and entity recognition to understand and respond to you! 🧠",
            "My name is NovaMind! I'm an intelligent chatbot built with Python, Flask, NLTK, and Scikit-learn. I'm here to help! 💡"
        ]
    },
    "capabilities": {
        "patterns": [
            "what can you do", "help me", "features", "abilities",
            "how can you help", "what do you know", "commands",
            "what are your skills", "capabilities"
        ],
        "responses": [
            "Here's what I can do:\n\n🔹 **Chat** — Have natural conversations\n🔹 **Analyze Sentiment** — Tell if text is positive, negative, or neutral\n🔹 **Answer Questions** — About science, tech, math, and more\n🔹 **Detect Entities** — Find names, dates, numbers in text\n🔹 **Remember Context** — I keep track of our conversation\n🔹 **Tell Jokes** — Need a laugh? Just ask!\n\nTry asking me anything! 🚀"
        ]
    },
    "thanks": {
        "patterns": [
            "thank you", "thanks", "thx", "appreciate it",
            "thanks a lot", "thank you so much", "grateful"
        ],
        "responses": [
            "You're welcome! Happy to help! 😊",
            "Anytime! That's what I'm here for! 🌟",
            "Glad I could help! Feel free to ask more! 💡",
            "No problem at all! 🤗"
        ]
    },
    "jokes": {
        "patterns": [
            "tell me a joke", "joke", "make me laugh",
            "funny", "humor", "tell me something funny"
        ],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛😄",
            "Why was the JavaScript developer sad? Because he didn't Node how to Express himself! 😂",
            "What's a computer's favorite snack? Microchips! 🍟💻",
            "Why do Python programmers have low self-esteem? Because they're constantly comparing themselves to others! 🐍😅",
            "What did the router say to the doctor? 'It hurts when IP!' 🏥😆",
            "How many programmers does it take to change a light bulb? None. That's a hardware problem! 💡😂"
        ]
    },
    "time": {
        "patterns": [
            "what time is it", "current time", "what's the time",
            "time now", "tell me the time", "clock"
        ],
        "responses": [
            f"The current time is {datetime.now().strftime('%I:%M %p')} ⏰",
        ]
    },
    "date": {
        "patterns": [
            "what's the date", "today's date", "what day is it",
            "current date", "date today", "what's today"
        ],
        "responses": [
            f"Today is {datetime.now().strftime('%A, %B %d, %Y')} 📅",
        ]
    },
    "programming": {
        "patterns": [
            "python", "java", "javascript", "programming",
            "coding", "software", "developer", "code",
            "c++", "machine learning", "ai", "artificial intelligence",
            "deep learning", "neural network", "algorithm"
        ],
        "responses": [
            "Great topic! Programming is the art of telling computers what to do. Popular languages include Python, Java, C++, and JavaScript. Each has its strengths:\n\n🐍 **Python** — AI/ML, data science, scripting\n☕ **Java** — Enterprise apps, Android\n⚡ **C++** — Systems programming, games\n🌐 **JavaScript** — Web development\n\nWhat would you like to know more about? 💻",
            "I love talking about tech! As an AI built with Python, I'm a bit biased towards it 😄. Are you interested in a specific language or concept?",
        ]
    },
    "science": {
        "patterns": [
            "science", "physics", "chemistry", "biology",
            "space", "universe", "planet", "atom",
            "molecule", "evolution", "dna", "cell",
            "quantum", "relativity", "gravity"
        ],
        "responses": [
            "Science is fascinating! Here are some mind-blowing facts:\n\n🌌 The observable universe is 93 billion light-years across\n⚛️ Atoms are 99.9% empty space\n🧬 Human DNA is 99.9% identical across all people\n🌊 Oceans contain about 20 million tons of gold\n\nWhat scientific topic interests you? 🔬",
            "Great question! Science helps us understand the universe. From quantum mechanics to astrophysics, there's always something new to discover! What area are you curious about? 🧪"
        ]
    },
    "math": {
        "patterns": [
            "math", "mathematics", "calculate", "equation",
            "algebra", "calculus", "geometry", "statistics",
            "probability", "number", "formula"
        ],
        "responses": [
            "Mathematics is the language of the universe! 🔢\n\nFun math facts:\n• π has been calculated to over 100 trillion digits\n• A googol is 10^100 (that's where 'Google' got its name!)\n• Zero is the only number that can't be represented in Roman numerals\n\nNeed help with a specific math concept? 📐",
        ]
    },
    "motivation": {
        "patterns": [
            "motivate me", "inspire me", "motivation",
            "i'm sad", "feeling down", "cheer me up",
            "encouragement", "i need help", "feeling bad"
        ],
        "responses": [
            "Remember: Every expert was once a beginner. You've got this! 💪🌟",
            "The only way to do great work is to love what you do. Keep pushing forward! 🚀",
            "Believe in yourself! Every line of code you write is one step closer to mastery. 💻✨",
            "Difficult roads often lead to beautiful destinations. Stay strong! 🏔️🌈",
            "You're doing better than you think. Progress isn't always visible, but it's always happening! 🌱"
        ]
    },
    "creator": {
        "patterns": [
            "who made you", "who created you", "who built you",
            "your creator", "developer", "who is your developer",
            "who programmed you"
        ],
        "responses": [
            "I was created by **Khaled Noaman**, a Computer Engineering student at Istanbul Arel University! He built me using Python, Flask, NLTK, and Scikit-learn. Check out his work at github.com/KhaledNoaman 🚀",
        ]
    }
}


# ============================================
# NLP Engine
# ============================================
class NLPEngine:
    """Natural Language Processing engine for the chatbot."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()
        self.vectorizer = TfidfVectorizer()
        self._build_intent_model()
    
    def _build_intent_model(self):
        """Build TF-IDF model from knowledge base patterns."""
        self.intent_labels = []
        self.intent_patterns = []
        
        for intent, data in KNOWLEDGE_BASE.items():
            for pattern in data["patterns"]:
                self.intent_labels.append(intent)
                self.intent_patterns.append(pattern.lower())
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.intent_patterns)
    
    def preprocess(self, text):
        """Tokenize, lowercase, remove stopwords, and lemmatize."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens 
                  if t not in self.stop_words and len(t) > 1]
        
        return ' '.join(tokens)
    
    def classify_intent(self, text):
        """Classify user message intent using TF-IDF + cosine similarity."""
        processed = self.preprocess(text)
        if not processed:
            processed = text.lower()
        
        user_vector = self.vectorizer.transform([processed])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score > 0.15:
            return self.intent_labels[best_idx], best_score
        
        return None, 0.0
    
    def analyze_sentiment(self, text):
        """Simple sentiment analysis using keyword matching."""
        positive_words = {
            'good', 'great', 'awesome', 'excellent', 'amazing', 'wonderful',
            'fantastic', 'love', 'happy', 'joy', 'beautiful', 'brilliant',
            'perfect', 'best', 'nice', 'cool', 'super', 'like', 'enjoy',
            'thank', 'thanks', 'helpful', 'impressive', 'outstanding'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'sad',
            'angry', 'worst', 'ugly', 'stupid', 'boring', 'annoying',
            'disappointed', 'frustrating', 'difficult', 'problem', 'wrong',
            'fail', 'error', 'broken', 'useless', 'poor'
        }
        
        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        
        if pos_count > neg_count:
            score = min(pos_count * 0.3, 1.0)
            return "positive", score
        elif neg_count > pos_count:
            score = min(neg_count * 0.3, 1.0)
            return "negative", score
        else:
            return "neutral", 0.0
    
    def extract_entities(self, text):
        """Extract named entities (simple rule-based)."""
        entities = []
        
        # Numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        for n in numbers:
            entities.append({"type": "NUMBER", "value": n})
        
        # Emails
        emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
        for e in emails:
            entities.append({"type": "EMAIL", "value": e})
        
        # URLs
        urls = re.findall(r'https?://\S+', text)
        for u in urls:
            entities.append({"type": "URL", "value": u})
        
        # Dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        for d in dates:
            entities.append({"type": "DATE", "value": d})
        
        # Capitalized words (potential names/places)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
        common_words = {'I', 'The', 'This', 'That', 'What', 'How', 'Why', 
                       'When', 'Where', 'Who', 'Can', 'Could', 'Would',
                       'Should', 'Hello', 'Hi', 'Hey', 'Please', 'Thanks',
                       'Thank', 'Yes', 'No', 'Ok', 'True', 'False'}
        for c in caps:
            if c not in common_words:
                entities.append({"type": "PROPER_NOUN", "value": c})
        
        return entities


# ============================================
# Conversation Manager
# ============================================
class ConversationManager:
    """Manages conversation context and history."""
    
    def __init__(self):
        self.nlp = NLPEngine()
        self.conversations = {}
    
    def get_response(self, user_message, session_id="default"):
        """Generate a response for the user message."""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "history": [],
                "message_count": 0,
                "start_time": datetime.now().isoformat()
            }
        
        context = self.conversations[session_id]
        context["message_count"] += 1
        
        # Analyze the message
        intent, confidence = self.nlp.classify_intent(user_message)
        sentiment, sent_score = self.nlp.analyze_sentiment(user_message)
        entities = self.nlp.extract_entities(user_message)
        
        # Generate response
        if intent and intent in KNOWLEDGE_BASE:
            response = random.choice(KNOWLEDGE_BASE[intent]["responses"])
            # Handle dynamic time/date
            if intent == "time":
                response = f"The current time is {datetime.now().strftime('%I:%M %p')} ⏰"
            elif intent == "date":
                response = f"Today is {datetime.now().strftime('%A, %B %d, %Y')} 📅"
        else:
            response = self._generate_fallback(user_message)
        
        # Store in history
        context["history"].append({
            "user": user_message,
            "bot": response,
            "intent": intent,
            "confidence": float(confidence) if confidence else 0,
            "sentiment": sentiment,
            "entities": entities,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "response": response,
            "intent": intent or "unknown",
            "confidence": round(float(confidence), 3) if confidence else 0,
            "sentiment": {
                "label": sentiment,
                "score": round(sent_score, 3)
            },
            "entities": entities,
            "message_count": context["message_count"]
        }
    
    def _generate_fallback(self, message):
        """Generate a response when no intent is matched."""
        fallbacks = [
            "That's an interesting question! While I'm still learning, I'd love to explore that topic with you. Could you tell me more? 🤔",
            "Hmm, I'm not sure about that one, but I'm always learning! Try asking me about science, tech, programming, or just say hi! 💡",
            "Great question! I don't have a specific answer for that yet, but I can help with programming, science, math, and general conversation. What else would you like to know? 🌟",
            "I'm still expanding my knowledge base! In the meantime, try asking me to tell you a joke, analyze sentiment, or chat about technology! 🤖",
        ]
        return random.choice(fallbacks)
    
    def get_stats(self, session_id="default"):
        """Get conversation statistics."""
        if session_id not in self.conversations:
            return {"message_count": 0, "history": []}
        
        context = self.conversations[session_id]
        sentiments = [h["sentiment"] for h in context["history"]]
        intents = [h["intent"] for h in context["history"] if h["intent"]]
        
        return {
            "message_count": context["message_count"],
            "start_time": context["start_time"],
            "sentiment_summary": {
                "positive": sentiments.count("positive"),
                "negative": sentiments.count("negative"),
                "neutral": sentiments.count("neutral")
            },
            "top_intents": list(set(intents)),
            "total_entities": sum(len(h.get("entities", [])) for h in context["history"])
        }


# ============================================
# Flask Routes
# ============================================
chatbot = ConversationManager()


@app.route('/')
def home():
    """Serve the chat interface."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
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
    """Get conversation statistics."""
    session_id = session.get('session_id', 'default')
    return jsonify(chatbot.get_stats(session_id))


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze text without chatting."""
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
        "intent": {"label": intent or "unknown", "confidence": round(float(confidence), 3)},
        "word_count": len(text.split()),
        "char_count": len(text)
    })


# ============================================
# Main
# ============================================
if __name__ == '__main__':
    print("=" * 50)
    print("  NovaMind AI Assistant")
    print("  Starting server...")
    print("=" * 50)
    print()
    print("  Open in your browser:")
    print("  http://localhost:5000")
    print()
    print("=" * 50)
    app.run(debug=True, port=5000)
