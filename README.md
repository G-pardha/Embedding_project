# 🔍 Semantic Search Engine

A semantic search engine built from scratch using **Sentence Transformers** and **ChromaDB**. Unlike traditional keyword search, this engine understands the *meaning* behind your queries and finds the most relevant results.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange?style=flat)
![SentenceTransformers](https://img.shields.io/badge/Sentence_Transformers-AI-green?style=flat)

## 🧠 How It Works

```
Your Query → AI Model → Vector (numbers) → Compare with stored vectors → Best matches!
```

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  "what is    │     │  SentenceTransf  │     │  ChromaDB    │
│   machine    │────▶│  ormer Model     │────▶│  Vector DB   │
│   learning"  │     │  (encode to vec) │     │  (find match)│
└──────────────┘     └─────────────────┘     └──────┬───────┘
                                                     │
                                              ┌──────▼───────┐
                                              │  Top 3 most  │
                                              │  similar docs│
                                              └──────────────┘
```

### Key Concepts
- **Embeddings**: Text converted into numerical vectors (lists of numbers) that capture meaning
- **Vector Database**: Stores these vectors and finds similar ones using cosine similarity
- **Semantic Search**: Matches by *meaning*, not exact keywords — so "coding" finds results about "programming"

## 📁 Project Structure

```
Embedding_project/
├── ingest.py              # Load data into ChromaDB
├── search.py              # Interactive search loop
├── dataset.json           # 500 documents across 30+ categories
├── dataset.txt            # Alternative text dataset (544 paragraphs)
├── questions.txt          # 200+ sample questions to test
├── requirements.txt       # Python dependencies
├── PSEUDOCODE_CHEATSHEET.md  # Quick reference for the pipeline
├── Chroma_db/             # ChromaDB storage (auto-generated)
└── venv/                  # Virtual environment (not pushed)
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Embedding_project.git
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

### 4. Ingest data (store documents as vectors)
```bash
python ingest.py
```
This will:
- Load the AI model (`all-MiniLM-L6-V2`)
- Read 500 documents from `dataset.json`
- Convert them to embeddings (vectors)
- Store them in ChromaDB with metadata (title, category)

### 5. Search!
```bash
python search.py
```
Type your query and get results! Type `quit` to exit.

## 💡 Example Usage

```
search: how does machine learning work
  [AI] Machine Learning
  Machine learning is a branch of artificial intelligence that enables
  computers to learn from data without being explicitly programmed.
  distance: 0.4521

  [AI] Deep Learning
  Deep learning uses neural networks with many layers to model complex
  patterns in data.
  distance: 0.7832

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
| 🤖 AI | Machine Learning, LLMs, RAG, Embeddings, GPT |
| 💻 Programming | Python, JavaScript, Git, APIs, Design Patterns |
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
| + more... | Education, Lifestyle, Technology, Culture |

## 🔧 How the Code Works

### `ingest.py` — Data Pipeline
```python
# 1. Load the AI model
model = SentenceTransformer("all-MiniLM-L6-V2")

# 2. Connect to ChromaDB
client = chromadb.PersistentClient(path="./Chroma_db")

# 3. Read documents from JSON
data = json.load(open("dataset.json"))

# 4. Convert text → vectors
embeddings = model.encode(documents).tolist()

# 5. Store in database
collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
```

### `search.py` — Query Loop
```python
# 1. User types a query
query = input("search: ")

# 2. Convert query to vector
query_vec = model.encode(query).tolist()

# 3. Find similar vectors in database
results = collection.query(query_embeddings=[query_vec], n_results=3)

# 4. Show results with distance < 1.0 (only relevant matches)
for doc, distance, metadata in zip(...):
    if distance < 1.0:
        print(f"[{metadata['category']}] {metadata['title']}")
```

## 📐 Understanding Distance

| Distance | Meaning |
|---|---|
| `0.0 - 0.5` | 🟢 Very strong match |
| `0.5 - 0.8` | 🟡 Good match |
| `0.8 - 1.0` | 🟠 Weak but relevant |
| `> 1.0` | 🔴 Not relevant (filtered out) |

## 🛠️ Tech Stack

- **[Sentence Transformers](https://www.sbert.net/)** — AI model to convert text into embeddings
- **[ChromaDB](https://www.trychroma.com/)** — Open-source vector database for similarity search
- **[all-MiniLM-L6-V2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)** — Lightweight, fast embedding model (384 dimensions)
- **Python 3.8+**

## 🤔 Limitations

- This is a **document finder**, not a chatbot — it finds similar text, it doesn't "think"
- Queries like *"suggest me a movie"* won't work because they need reasoning
- Results are only as good as the data in the database
- The model works best with English text

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ **If you found this helpful, give it a star!**
