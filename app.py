import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import json
import os


# ---- Page Config ----
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)


# ---- Load model and database (cached so it only loads once) ----
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-V2")

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="./Chroma_db")
    collection = client.get_or_create_collection(name="Documents")
    
    # Auto-ingest if collection is empty (needed for cloud deployment)
    if collection.count() == 0:
        dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = [item["text"] for item in data]
        ids = [item["id"] for item in data]
        metadatas = [{"title": item["title"], "category": item["category"]} for item in data]
        
        model = load_model()
        embeddings = model.encode(documents, show_progress_bar=True).tolist()
        collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
    
    return collection

model = load_model()
collection = load_collection()


# ---- Custom CSS for styling ----
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .result-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #667eea;
    }
    .result-category {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .result-text {
        color: #ccc;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .distance-badge {
        color: #888;
        font-size: 0.8rem;
        margin-top: 8px;
    }
    .stats-box {
        text-align: center;
        padding: 1rem;
        background: #1a1a2e;
        border-radius: 10px;
    }
    .stats-number {
        font-size: 2rem;
        font-weight: 800;
        color: #667eea;
    }
    .stats-label {
        color: #888;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ---- Header ----
st.markdown('<h1 class="main-title">🔍 Semantic Search Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Search by meaning, not just keywords — powered by AI embeddings & ChromaDB</p>', unsafe_allow_html=True)


# ---- Stats row ----
doc_count = collection.count()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="stats-box"><div class="stats-number">{doc_count}</div><div class="stats-label">Documents Indexed</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stats-box"><div class="stats-number">384</div><div class="stats-label">Vector Dimensions</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stats-box"><div class="stats-number">30+</div><div class="stats-label">Categories</div></div>', unsafe_allow_html=True)

st.markdown("---")


# ---- Search Bar ----
col_search, col_btn = st.columns([5, 1])
with col_search:
    query = st.text_input("Search", placeholder="Ask anything... e.g. 'how does machine learning work'", label_visibility="collapsed")
with col_btn:
    search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")


# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    n_results = st.slider("Number of results", 1, 10, 5)
    threshold = st.slider("Distance threshold", 0.5, 1.5, 1.0, 0.05, help="Lower = stricter matches, Higher = more results")
    
    st.markdown("---")
    st.markdown("### 📂 Categories")
    
    # Get all unique categories
    all_results = collection.get(include=["metadatas"])
    categories = sorted(set(m["category"] for m in all_results["metadatas"] if m is not None and "category" in m))
    
    selected_category = st.selectbox("Filter by category", ["All"] + categories)
    
    st.markdown("---")
    st.markdown("### 💡 Try These")
    sample_queries = [
        "how does machine learning work",
        "best places to visit in India",
        "what is blockchain technology",
        "healthy breakfast ideas",
        "famous horror movies",
        "how to manage stress",
        "what is quantum computing",
        "cricket world cup",
    ]
    for sq in sample_queries:
        if st.button(sq, key=sq, use_container_width=True):
            st.session_state["auto_query"] = sq
            st.rerun()


# ---- Handle auto-query from sidebar ----
if "auto_query" in st.session_state:
    query = st.session_state.pop("auto_query")


# ---- Search Logic ----
if query:
    # Encode the query
    query_vec = model.encode(query).tolist()
    
    # Build query params
    query_params = {
        "query_embeddings": [query_vec],
        "n_results": n_results,
        "include": ["documents", "distances", "metadatas"]
    }
    
    # Add category filter if selected
    if selected_category != "All":
        query_params["where"] = {"category": selected_category}
    
    results = collection.query(**query_params)
    
    # Filter by threshold and display
    found = False
    st.markdown(f"### Results for: *\"{query}\"*")
    
    for doc, distance, metadata in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        if distance < threshold:
            found = True
            
            # Color based on distance
            if distance < 0.5:
                color = "#4CAF50"  # green
                strength = "🟢 Excellent match"
            elif distance < 0.8:
                color = "#FFC107"  # yellow
                strength = "🟡 Good match"
            else:
                color = "#FF9800"  # orange
                strength = "🟠 Weak match"
            
            st.markdown(f"""
            <div class="result-card" style="border-left-color: {color};">
                <span class="result-category">{metadata.get('category', 'N/A')}</span>
                <div class="result-title">{metadata.get('title', 'Untitled')}</div>
                <div class="result-text">{doc}</div>
                <div class="distance-badge">{strength} · distance: {distance:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    if not found:
        st.warning("😕 No relevant results found! Try a different query or increase the distance threshold.")

elif search_clicked:
    st.info("Please type something in the search box!")
