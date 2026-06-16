from sentence_transformers import SentenceTransformer
import chromadb

#load the model 
model=SentenceTransformer("all-MiniLM-L6-V2")

#load the database

client=chromadb.PersistentClient(path="./Chroma_db")

collection=client.get_or_create_collection(name = "Documents")



#search loop

while True:
    query=input("search: ")
    if query=="quit":
        break
    query_vec=model.encode(query).tolist()
    results=collection.query(query_embeddings=[query_vec],n_results=3)

    #only show close matches (distance < 1.0)
    found = False
    for doc, distance, metadata in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        if distance < 1.0:
            print(f"  [{metadata['category']}] {metadata['title']}")
            print(f"  {doc}")
            print(f"  distance: {distance:.4f}")
            print()
            found = True

    if not found:
        print("No relevant results found!")
