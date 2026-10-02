import os
import json
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CACHE_DIR = "rag_cache"
DATASET_PATH = "schemes_rag_dataset.jsonl"
EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "scheme_embeddings.npy")
DOCS_FILE = os.path.join(CACHE_DIR, "schemes_store.json")
PROGRESS_FILE = os.path.join(CACHE_DIR, "indexing_progress.json")

def create_embeddings_index(batch_size=500, model="text-embedding-3-small"):
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    client = OpenAI()
    
    print(f"Reading dataset from {DATASET_PATH}...")
    documents = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
                
    total_docs = len(documents)
    print(f"Total documents to index: {total_docs}")
    
    # Save documents store
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False)
    print(f"Saved documents store to {DOCS_FILE}")
    
    # Check if embeddings already exist completely
    if os.path.exists(EMBEDDINGS_FILE):
        existing_matrix = np.load(EMBEDDINGS_FILE)
        if len(existing_matrix) == total_docs:
            print(f"Embeddings matrix already exists with shape {existing_matrix.shape}. Indexing complete!")
            return existing_matrix, documents
            
    # Check partial progress
    start_idx = 0
    all_embeddings = []
    if os.path.exists(PROGRESS_FILE) and os.path.exists(EMBEDDINGS_FILE + ".tmp.npy"):
        try:
            with open(PROGRESS_FILE, "r") as pf:
                progress = json.load(pf)
                start_idx = progress.get("completed_count", 0)
                partial = np.load(EMBEDDINGS_FILE + ".tmp.npy")
                all_embeddings = list(partial)
                print(f"Resuming indexing from item {start_idx}/{total_docs}...")
        except Exception as e:
            print(f"Error loading checkpoint, starting fresh: {e}")
            start_idx = 0
            all_embeddings = []

    print(f"Generating OpenAI embeddings using '{model}' in batches of {batch_size}...")
    t0 = time.time()
    
    for i in range(start_idx, total_docs, batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_texts = [d["search_text"] for d in batch_docs]
        
        batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size
        print(f"[{time.strftime('%H:%M:%S')}] Embedding batch {batch_num}/{total_batches} (items {i} to {min(i + batch_size, total_docs)})...", end=" ", flush=True)
        
        retries = 5
        delay = 2
        for attempt in range(retries):
            try:
                b_t0 = time.time()
                resp = client.embeddings.create(
                    model=model,
                    input=batch_texts
                )
                b_time = time.time() - b_t0
                batch_vectors = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_vectors)
                print(f"Done in {b_time:.2f}s!")
                break
            except Exception as e:
                print(f"\nWarning: Batch {batch_num} failed (attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2
                
        # Save temporary progress checkpoint
        np.save(EMBEDDINGS_FILE + ".tmp.npy", np.array(all_embeddings, dtype=np.float32))
        with open(PROGRESS_FILE, "w") as pf:
            json.dump({"completed_count": len(all_embeddings)}, pf)
            
    embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
    # Normalize rows for fast cosine similarity dot products
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized_matrix = embeddings_matrix / norms
    
    np.save(EMBEDDINGS_FILE, normalized_matrix)
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    if os.path.exists(EMBEDDINGS_FILE + ".tmp.npy"):
        os.remove(EMBEDDINGS_FILE + ".tmp.npy")
        
    total_time = time.time() - t0
    print(f"Indexing successfully finished in {total_time:.2f}s!")
    print(f"Saved normalized embeddings matrix with shape {normalized_matrix.shape} to {EMBEDDINGS_FILE}")
    return normalized_matrix, documents

if __name__ == "__main__":
    create_embeddings_index()
