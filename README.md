# NovaMind - AI Assistant Chatbot

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Groq](https://img.shields.io/badge/Groq_AI-000000?style=for-the-badge&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **An intelligent conversational AI assistant powered by Groq (Llama 3 70B) with NLP-powered intent classification, sentiment analysis, named entity recognition, and a beautiful real-time web interface.**

---

## Overview

NovaMind is a full-stack AI chatbot built with Python and Flask. It uses **Groq's Llama 3 70B** model for intelligent responses and local **Natural Language Processing (NLP)** techniques to classify intents, analyze sentiment, extract entities, and provide real-time analytics. The web interface features a modern dark-themed design with live NLP analysis visualization.

## Key Features

| Feature | Description |
|---------|-------------|
| **AI-Powered Responses** | Groq API with Llama 3 70B for smart, contextual answers |
| **Sentiment Analysis** | Detects positive, negative, and neutral emotions in text |
| **Entity Recognition** | Extracts names, numbers, emails, URLs, dates from messages |
| **Context Memory** | Remembers conversation history for contextual responses |
| **Real-time Analytics** | Live stats: message count, sentiment distribution, entity count |
| **Beautiful Web UI** | Dark-themed glassmorphism design with smooth animations |
| **REST API** | JSON API endpoints for chat, analysis, and statistics |
| **Expandable Knowledge Base** | Easy to add new topics and responses |

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.9+ | Core language |
| Flask | Web server and REST API |
| Groq API | Llama 3 70B language model |
| NLTK | Tokenization, lemmatization, stopword removal |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| NumPy | Numerical computations |
| HTML/CSS/JS | Beautiful responsive web interface |

## Architecture

```
+---------------------------------------------+
|           NovaMind Web Interface              |
|  +--------+  +-----------+  +-------------+  |
|  | Chat   |  | NLP       |  | Analytics   |  |
|  | Window |  | Analysis  |  | Dashboard   |  |
|  +--------+  +-----+-----+  +-------------+  |
+--------------------|------------------------+
                     | REST API
+--------------------|-----------------------+
|            Flask Server                     |
|  +-----------------------------------------+|
|  |      Conversation Manager               ||
|  |  +----------------+ +----------------+  ||
|  |  | NLP Engine     | | Groq AI Client |  ||
|  |  | (Local)        | | (Llama 3 70B)  |  ||
|  |  | - Intent       | | - Smart Reply  |  ||
|  |  | - Sentiment    | | - Context      |  ||
|  |  | - Entities     | | - Memory       |  ||
|  |  +----------------+ +----------------+  ||
|  +-----------------------------------------+|
|  +-------------------+                      |
|  | Knowledge Base    |                      |
|  | (Fallback)        |                      |
|  +-------------------+                      |
+---------------------------------------------+
```

## Getting Started

### Prerequisites
```bash
Python >= 3.8
Groq API key (free at console.groq.com/keys)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/KHALEDNOAMAN/NovaMind-Chatbot.git
cd NovaMind-Chatbot

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key to .env file
echo GROQ_API_KEY=your_key_here > .env

# Run the chatbot
python app.py
```

Then open **http://localhost:5000**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/api/chat` | POST | Send message, get AI response |
| `/api/stats` | GET | Conversation statistics |
| `/api/analyze` | POST | Analyze text (sentiment, entities) |

### Example Request
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'
```

## Project Structure

```
NovaMind-Chatbot/
|-- app.py              # Main application
|-- requirements.txt    # Python dependencies
|-- .env               # API key (not committed)
|-- templates/
|   +-- index.html     # Chat interface
+-- README.md
```

## License

This project is licensed under the MIT License.

## Author

**Khaled Noaman** - Computer Engineering Student at Istanbul Arel University

- [GitHub](https://github.com/KhaledNoaman)
- [LinkedIn](https://www.linkedin.com/in/khalednoaman1/)
