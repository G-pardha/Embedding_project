from sentence_transformers import SentenceTransformer
import chromadb
import json


#load the model
model=SentenceTransformer("all-MiniLM-L6-V2")

#created a database
client=chromadb.PersistentClient(path= "./Chroma_db")

#delete old collection if it exists (fresh start)
try:
    client.delete_collection(name="Documents")
except:
    pass

#creating a collection to store our embeddings
collection= client.get_or_create_collection(name = "Documents")

#read data from JSON file
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

#extract documents, ids, and metadata from JSON
documents = [item["text"] for item in data]
ids = [item["id"] for item in data]
metadatas = [{"title": item["title"], "category": item["category"]} for item in data]

print(f"loaded {len(documents)} documents from dataset.json")

#convert to embeddings
embeddings=model.encode(documents, show_progress_bar=True).tolist()

#store in the collection with metadata
collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)

#after successfully storing we get this message
print(f"successfully stored {collection.count()} documents.")