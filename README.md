# SENTINEL - Silence Index Investigation System

A forensic AI-powered platform for detecting and analyzing institutional bias in civic complaint systems. SENTINEL uncovers patterns of systemic neglect by analyzing which complaints get "silenced" - delayed, ignored, or deprioritized based on demographic factors.

## Overview

SENTINEL investigates bias patterns across:
- **Demographics**: Gender, caste, and income-based discrimination
- **Geography**: Ward-level disparities in complaint resolution
- **Categories**: Which complaint types face systematic neglect
- **Temporal**: How complaints decay and get forgotten over time

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.13, Flask |
| Vector Database | Qdrant |
| AI/LLM | Google Gemini 2.5 Flash |
| Embeddings | Sentence-Transformers (BAAI/bge-small-en-v1.5) |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js |

## Features

- Real-time conversational AI analysis with SENTINEL agent
- Dynamic chart generation from natural language queries
- Semantic search across 10,000+ complaint records
- Demographic bias detection and quantification
- Geographic hotspot identification
- Temporal decay analysis
- Chat history persistence in Qdrant

## Project Structure

```
sentinel/
├── app.py                 # Flask API server (13 endpoints)
├── chat_agent.py          # Conversational AI with Qdrant memory
├── queries.py             # Qdrant query functions
├── agentic_ai.py          # Autonomous bias investigation
├── multimodal.py          # Image processing capabilities
├── generate_data.py       # Synthetic data generation
├── generate_embeddings.py # Vector embedding pipeline
├── index_to_qdrant.py     # Qdrant indexing
├── static/
│   ├── index.html         # Main dashboard
│   ├── css/style.css      # Forensic dark theme
│   └── js/script.js       # Frontend logic
└── data/                  # CSV and embeddings (gitignored)
```

## Installation

### Prerequisites

- Python 3.10+
- Qdrant (local or cloud)
- Google AI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/arduvey29/Sentinel.git
cd Sentinel
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install flask flask-cors qdrant-client sentence-transformers langchain-google-genai python-dotenv pandas numpy
```

4. Configure environment:
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

5. Start Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

6. Generate data and index:
```bash
python generate_data.py
python generate_embeddings.py
python index_to_qdrant.py
```

7. Run the application:
```bash
python app.py
```

8. Open browser at `http://localhost:5000`

## API Endpoints

### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stats` | Overall statistics |
| GET | `/api/demographic-silence` | Demographic breakdown |
| GET | `/api/geographic-silence` | Ward-level analysis |
| GET | `/api/complaint-types` | Category analysis |
| GET | `/api/temporal-decay` | Time-based patterns |
| POST | `/api/search` | Semantic search |

### AI Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send message to SENTINEL |
| GET | `/api/chat/sessions` | List chat sessions |
| GET | `/api/chat/history/<id>` | Get session history |
| POST | `/api/chat/session/new` | Create new session |
| POST | `/api/chat/search` | Search chat history |

## Usage Examples

### Chat Queries
```
"What is the overall silence rate?"
"Show me income-based bias with a chart"
"Which wards have the highest silence rates?"
"Analyze caste-based discrimination patterns"
"Show temporal decay - how complaints get ignored over time"
```

### API Request
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the silence rate?", "session_id": null}'
```

## The Silence Score

Each complaint receives a Silence Score (0-100) based on:
- Days pending without resolution
- Number of follow-ups ignored
- Escalation attempts dismissed
- Category-based historical neglect rates

Scores above 70 indicate systematic silencing.

## Key Findings

SENTINEL has uncovered significant bias patterns:
- Lower income brackets face 2.3x higher silencing rates
- Certain wards show 40%+ complaint neglect
- Specific complaint categories are systematically deprioritized
- Complaints decay rapidly after 30 days without action

## License

MIT License

## Version

v1.0.0

---

Built for investigating institutional bias in civic systems.
