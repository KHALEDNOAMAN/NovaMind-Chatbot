# 🧠 NovaMind — AI Assistant Chatbot

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-000000?style=for-the-badge&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **An intelligent conversational AI assistant powered by Groq (Llama 3 70B) with NLP-powered intent classification, sentiment analysis, named entity recognition, and a beautiful real-time web interface.**

---

## 🎯 Overview

NovaMind is a full-stack AI chatbot built with Python and Flask. It uses **Groq's Llama 3 70B** model for intelligent responses and local **Natural Language Processing (NLP)** techniques to classify intents, analyze sentiment, extract entities, and provide real-time analytics. The web interface features a modern dark-themed design with live NLP analysis visualization.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI-Powered Responses** | Groq API with Llama 3 70B for smart, contextual answers |
| 😊 **Sentiment Analysis** | Detects positive, negative, and neutral emotions in text |
| 🔍 **Entity Recognition** | Extracts names, numbers, emails, URLs, dates from messages |
| 🧠 **Context Memory** | Remembers conversation history for contextual responses |
| 📊 **Real-time Analytics** | Live stats: message count, sentiment distribution, entity count |
| 🎨 **Beautiful Web UI** | Dark-themed glassmorphism design with smooth animations |
| 🔌 **REST API** | JSON API endpoints for chat, analysis, and statistics |
| 📚 **Expandable Knowledge Base** | Easy to add new topics and responses |

## 🖥️ Screenshots

The chatbot features a 3-panel layout:
- **Left Panel** — Conversation stats and feature list
- **Center** — Chat interface with message bubbles and typing indicators
- **Right Panel** — Real-time NLP analysis (intent, sentiment, entities, confidence)

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.9+ | Core language |
| Flask | Web server & REST API |
| Groq API | Llama 3 70B language model |
| NLTK | Tokenization, lemmatization, stopword removal |
| Scikit-learn | TF-IDF vectorization & cosine similarity |
| NumPy | Numerical computations |
| HTML/CSS/JS | Beautiful responsive web interface |

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                  Web Interface                    │
│   ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│   │  Stats   │  │   Chat   │  │  NLP Panel   │  │
│   │  Panel   │  │  Window  │  │  (Analysis)  │  │
│   └──────────┘  └────┬─────┘  └──────────────┘  │
└───────────────────────┼──────────────────────────┘
                        │ REST API
┌───────────────────────┼──────────────────────────┐
│                 Flask Server                      │
│  ┌────────────────────┼──────────────────────┐   │
│  │          Conversation Manager              │   │
│  │  ┌─────────┐  ┌─────────┐  ┌───────────┐ │   │
│  │  │ Intent  │  │Sentiment│  │  Entity   │ │   │
│  │  │Classify │  │Analysis │  │Recognition│ │   │
│  │  └────┬────┘  └────┬────┘  └─────┬─────┘ │   │
│  │       └─────────┬──┘             │        │   │
│  │            NLP Engine            │        │   │
│  │     (TF-IDF + Cosine Sim)       │        │   │
│  └──────────────────────────────────┘        │   │
│                                              │   │
│  ┌──────────────────────────────────┐        │   │
│  │        Knowledge Base            │        │   │
│  │  (12+ categories, 50+ patterns) │        │   │
│  └──────────────────────────────────┘        │   │
└──────────────────────────────────────────────┘
```

## 🚀 Getting Started

### Prerequisites
```bash
Python >= 3.8
```

### Installation & Run
```bash
# Clone the repository
git clone https://github.com/KHALEDNOAMAN/NovaMind-Chatbot.git
cd NovaMind-Chatbot

# Install dependencies
pip install -r requirements.txt

# Run the chatbot
python app.py
```

Then open **http://localhost:5000** in your browser! 🌐

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web chat interface |
| `/api/chat` | POST | Send message, get AI response |
| `/api/stats` | GET | Conversation statistics |
| `/api/analyze` | POST | Analyze text (sentiment, entities, intent) |

### Example API Call:
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a joke"}'
```

### Response:
```json
{
  "response": "Why do programmers prefer dark mode? Because light attracts bugs! 🐛😄",
  "intent": "jokes",
  "confidence": 0.892,
  "sentiment": { "label": "neutral", "score": 0.0 },
  "entities": [],
  "message_count": 1
}
```

## 📁 Project Structure

```
NovaMind-Chatbot/
├── app.py                  # Main application (Flask + NLP engine)
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Beautiful chat web interface
├── LICENSE
└── README.md
```

## 🔬 NLP Techniques Used

- **TF-IDF Vectorization** — Converts text to numerical features based on word importance
- **Cosine Similarity** — Measures similarity between user input and known patterns
- **Tokenization** — Breaks text into individual words (NLTK)
- **Lemmatization** — Reduces words to base form ("running" → "run")
- **Stopword Removal** — Filters out common words ("the", "is", "at")
- **Named Entity Recognition** — Regex-based extraction of emails, URLs, numbers, dates

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Khaled Noaman** — Computer Engineering Student at Istanbul Arel University

- [GitHub](https://github.com/KhaledNoaman)
- [LinkedIn](https://www.linkedin.com/in/khalednoaman1/)
