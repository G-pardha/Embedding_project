# 📋 Pseudocode Cheatsheet — Embeddings & Vector Database Pipeline

---

## STEP 1: Load the Embedding Model

```python
# Import the library
from sentence_transformers import SentenceTransformer

# Load a pre-trained model (downloads ~80MB first time, cached after)
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**What this does:** Loads a model that can convert any text → 384 numbers (vector).

---

## STEP 2: Convert Text → Vector (Embedding)

### Single text:
```python
text = "Python is a great programming language"

# encode() takes text and returns a numpy array of 384 numbers
vector = model.encode(text)

# vector = [ 0.0123, -0.0456, 0.0789, ..., -0.0321 ]   (384 numbers)
# vector.shape = (384,)
# type(vector) = numpy.ndarray
```

### Multiple texts at once (faster!):
```python
texts = ["I love dogs", "The weather is sunny", "Python is fun"]

# encode() a list → returns a matrix of shape (3, 384)
vectors = model.encode(texts)

# vectors.shape = (3, 384)   → 3 texts, each with 384 numbers
# vectors[0] = embedding of "I love dogs"
# vectors[1] = embedding of "The weather is sunny"
# vectors[2] = embedding of "Python is fun"
```

---

## STEP 3: Compare Two Vectors (Cosine Similarity)

```python
import numpy as np

def cosine_similarity(v1, v2):
    # Step 1: Dot product → multiply matching elements and sum
    dot = np.dot(v1, v2)

    # Step 2: Magnitudes → length of each vector
    norm1 = np.linalg.norm(v1)    # = sqrt(v1[0]² + v1[1]² + ... + v1[383]²)
    norm2 = np.linalg.norm(v2)

    # Step 3: Divide
    return dot / (norm1 * norm2)
```

### Using it:
```python
vec1 = model.encode("I love dogs")
vec2 = model.encode("I adore puppies")
vec3 = model.encode("The stock market crashed")

similarity_1_2 = cosine_similarity(vec1, vec2)   # ~0.7+ (very similar!)
similarity_1_3 = cosine_similarity(vec1, vec3)   # ~0.05 (not related)
```

### Score meaning:
```
 1.0  = identical meaning
 0.7+ = very similar
 0.4+ = somewhat similar
 0.0  = no relation
-1.0  = opposite meaning
```

---

## STEP 4: Create a Vector Database (ChromaDB)

```python
import chromadb

# Create a ChromaDB client
# PersistentClient → saves data to disk (survives restarts)
# path = folder where data is stored
chroma_client = chromadb.PersistentClient(path="./chroma_db")
```

---

## STEP 5: Create a Collection (like a table)

```python
# get_or_create_collection:
#   - if "my_documents" exists → returns it
#   - if it doesn't exist → creates it
collection = chroma_client.get_or_create_collection(
    name="my_documents",
    metadata={"description": "My first vector database!"}
)
```

---

## STEP 6: Add Documents to the Collection

```python
# 1. Your documents (the actual text)
documents = [
    "Python is great for web development",
    "Machine learning can learn from data",
    "The solar system has eight planets",
]

# 2. Generate embeddings (text → vectors)
embeddings = model.encode(documents).tolist()   # .tolist() converts numpy → python list

# 3. Create unique IDs (required by ChromaDB)
ids = ["doc_0", "doc_1", "doc_2"]

# 4. Create metadata (optional, but useful for filtering)
metadatas = [
    {"category": "programming", "topic": "python"},
    {"category": "ai",          "topic": "ml"},
    {"category": "science",     "topic": "astronomy"},
]

# 5. Add everything to ChromaDB!
collection.add(
    documents=documents,       # original text (stored for retrieval)
    embeddings=embeddings,     # vector representations
    ids=ids,                   # unique identifiers
    metadatas=metadatas,       # extra info for filtering
)
```

---

## STEP 7: Search the Database (Query)

```python
# 1. User types a query
query = "What programming language should I learn?"

# 2. Convert query → vector using the SAME model
query_embedding = model.encode(query).tolist()

# 3. Search ChromaDB for top 3 most similar documents
results = collection.query(
    query_embeddings=[query_embedding],    # query vector (wrapped in a list)
    n_results=3,                           # how many results to return
)

# 4. results is a dictionary:
#    results['documents'][0]  → list of matched texts
#    results['distances'][0]  → list of distances (lower = more similar)
#    results['metadatas'][0]  → list of metadata dicts
#    results['ids'][0]        → list of document IDs

# 5. Display results
for rank, (doc, distance, metadata) in enumerate(zip(
    results['documents'][0],
    results['distances'][0],
    results['metadatas'][0],
), 1):
    similarity = 1 - distance     # convert distance → similarity score
    print(f"#{rank}  similarity: {similarity:.4f}")
    print(f"  category: {metadata['category']}")
    print(f"  text: {doc}")
```

---

## STEP 8: Search with Metadata Filter (Optional)

```python
# Search ONLY within documents where category = "programming"
filtered_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={"category": "programming"},     # filter by metadata!
)
```

---

## STEP 9: Connect to Existing Database (for a separate script)

```python
# Connect to the SAME database folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get the existing collection (NOT create — it already exists)
collection = chroma_client.get_collection("my_documents")

# Check how many documents are stored
count = collection.count()    # e.g., 12

# Get ALL documents
all_docs = collection.get()
# all_docs['documents']  → list of all texts
# all_docs['metadatas']  → list of all metadata
# all_docs['ids']        → list of all IDs
```

---

## 🔄 THE COMPLETE PIPELINE (Summary)

```
┌──────────────────────────────────────────────────────────────┐
│  PHASE 1: STORE                                              │
│                                                              │
│  model = SentenceTransformer('all-MiniLM-L6-v2')            │
│  embeddings = model.encode(documents).tolist()               │
│  client = chromadb.PersistentClient(path="./chroma_db")      │
│  collection = client.get_or_create_collection("my_docs")     │
│  collection.add(documents, embeddings, ids, metadatas)       │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  PHASE 2: SEARCH                                             │
│                                                              │
│  query_vec = model.encode("user question").tolist()           │
│  results = collection.query(                                 │
│      query_embeddings=[query_vec],                           │
│      n_results=3                                             │
│  )                                                           │
│  similarity = 1 - results['distances'][0][i]                 │
│  matched_text = results['documents'][0][i]                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 📦 Required Installs

```bash
pip install sentence-transformers chromadb
```

## 📁 Key Imports

```python
from sentence_transformers import SentenceTransformer   # for embeddings
import chromadb                                         # for vector database
import numpy as np                                      # for cosine similarity math
```
