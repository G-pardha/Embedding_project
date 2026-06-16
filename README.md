# 🔍 Semantic Search Engine

A semantic search engine built from scratch using **Sentence Transformers** and **ChromaDB** with a **Streamlit Web UI**. Unlike traditional keyword search, this engine understands the *meaning* behind your queries and finds the most relevant results.

### 🌐 [Live Demo → embedding-project.streamlit.app](https://embedding-project.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?style=flat&logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange?style=flat)
![SentenceTransformers](https://img.shields.io/badge/Sentence_Transformers-AI-green?style=flat)

## 🧠 How It Works

```
Your Query → AI Model → Vector (numbers) → Compare with stored vectors → Best matches!
```

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  "what is    │     │ SentenceTransf  │     │   ChromaDB   │
│   machine    │────▶│  ormer Model    │────▶│  Vector DB   │
│   learning"  │     │ (encode to vec) │     │ (find match) │
└──────────────┘     └─────────────────┘     └──────┬───────┘
                                                    │
                                             ┌──────▼───────┐
                                             │  Top results │
                                             │  with scores │
                                             └──────────────┘
```

### Key Concepts
- **Embeddings**: Text converted into numerical vectors (lists of numbers) that capture meaning
- **Vector Database**: Stores these vectors and finds similar ones using cosine similarity
- **Semantic Search**: Matches by *meaning*, not exact keywords — so "coding" finds results about "programming"

## ✨ Features

- 🔍 **Semantic Search** — Search by meaning, not keywords
- 🎨 **Streamlit Web UI** — Beautiful, interactive web interface
- 📂 **Category Filtering** — Filter results by 30+ categories
- 🎯 **Distance Threshold** — Adjustable relevance slider
- 📊 **Stats Dashboard** — Shows document count, vector dimensions
- 💡 **Sample Queries** — Click pre-built queries to test
- 🟢🟡🟠 **Color-coded Results** — See match quality at a glance
- 🚀 **Auto-Ingest** — App loads data automatically on first run

## 📁 Project Structure

```
Embedding_project/
├── app.py                 # Streamlit web UI (main app)
├── ingest.py              # CLI tool to load data into ChromaDB
├── search.py              # CLI tool for terminal-based search
├── dataset.json           # 500 documents across 30+ categories
├── questions.txt          # 200+ sample questions to test
├── requirements.txt       # Python dependencies
├── LEARN_EMBEDDINGS.md    # Learn how embeddings work
├── PSEUDOCODE_CHEATSHEET.md  # Quick reference for the pipeline
├── .gitignore             # Git ignore rules
├── Chroma_db/             # ChromaDB storage (auto-generated)
└── venv/                  # Virtual environment (not pushed)
```

| File | Purpose |
|---|---|
| `app.py` | 🌐 Streamlit web app with UI, filters, and auto-ingest |
| `ingest.py` | 🔧 Dev tool — manually reload/update data |
| `search.py` | 🔧 Dev tool — quick terminal search for testing |

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/G-pardha/Embedding_project.git
cd Embedding_project
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

The app will:
- Download the AI model (`all-MiniLM-L6-V2`) on first run
- Auto-ingest 500 documents from `dataset.json`
- Open at **http://localhost:8501**

> 💡 First launch takes ~2 min for model download + embedding. After that it's cached and instant!

### Alternative: CLI Search
```bash
# Ingest data manually
python ingest.py

# Search from terminal
python search.py
```

## 💡 Example Usage

### Web UI (Streamlit)
- Type queries in the search bar
- Adjust **number of results** and **distance threshold** in the sidebar
- Filter by **category** (AI, Sports, Movies, Health, etc.)
- Click **sample queries** in the sidebar to test

### CLI Search
```
search: how does machine learning work
  [AI] Machine Learning
  Machine learning is a branch of artificial intelligence that enables
  computers to learn from data without being explicitly programmed.
  distance: 0.4521

search: best places to visit in India
  [Travel] India Travel Guide
  India offers incredible diversity from the Himalayas to tropical beaches...
  distance: 0.6234

search: random gibberish text
  No relevant results found!

search: quit
```

## 📊 Dataset

The `dataset.json` contains **500 documents** across **30+ categories**:

| Category | Topics |
|---|---|
| 🤖 AI | Machine Learning, LLMs, RAG, Embeddings, Agents |
| 💻 Programming | Python, JavaScript, Rust, Git, APIs, Design Patterns |
| 🌐 Web | React, Node.js, Next.js, WebSockets, CSS |
| 🗄️ Database | SQL, MongoDB, Redis, PostgreSQL, Elasticsearch |
| ☁️ Cloud | Docker, Kubernetes, AWS, CI/CD, Serverless |
| 📊 Data Science | Pandas, NumPy, Visualization, Statistics |
| 🔬 Science | Quantum, DNA, Black Holes, Climate, Brain |
| 📜 History | WWII, Silk Road, Mughal Empire, Renaissance |
| ⚽ Sports | Cricket, Football, F1, Chess, Kabaddi |
| 🎬 Movies | Bollywood, Tollywood, Marvel, Nolan, K-Cinema |
| 🎵 Music | Beatles, K-Pop, Jazz, Indian Classical, Hip Hop |
| 💰 Finance | Stocks, Crypto, Budgeting, Retirement |
| 💪 Health | Sleep, Yoga, Nutrition, Mental Health |
| 🍳 Food | Indian, Japanese, Italian, Coffee, Baking |
| ✈️ Travel | India, Japan, Europe, Budget, Solo Travel |
| 🧠 Psychology | Biases, Growth Mindset, Memory, Motivation |
| 💼 Business | Startups, Marketing, E-commerce, Leadership |
| 🌍 Environment | Climate, Recycling, Electric Vehicles |
| 🐾 Animals | Marine Life, Endangered Species, Dinosaurs |
| 🎨 Creative | Photography, Music Production, UX Design |
| + more... | Education, Lifestyle, Technology, Culture, Gaming |

## 📐 Understanding Distance

| Distance | Meaning |
|---|---|
| `0.0 - 0.5` | 🟢 Excellent match |
| `0.5 - 0.8` | 🟡 Good match |
| `0.8 - 1.0` | 🟠 Weak but relevant |
| `> 1.0` | 🔴 Not relevant (filtered out) |

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** — Web UI framework for the search interface
- **[Sentence Transformers](https://www.sbert.net/)** — AI model to convert text into embeddings
- **[ChromaDB](https://www.trychroma.com/)** — Open-source vector database for similarity search
- **[all-MiniLM-L6-V2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)** — Lightweight, fast embedding model (384 dimensions)
- **Python 3.8+**

## 🌐 Deployment

### Streamlit Community Cloud (Free)
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Select this repo → `app.py` → Deploy!

The app auto-ingests data on first launch — no extra setup needed.

## 🤔 Limitations

- This is a **document finder**, not a chatbot — it retrieves similar text, it doesn't generate answers
- Results are only as good as the data in the database
- The model works best with English text
- First load takes ~2 min (model download + embedding creation)

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ **If you found this helpful, give it a star!**
