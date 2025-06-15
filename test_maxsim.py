import faiss
import numpy as np

def test_maxsim():
    """Test MaxSim implementation"""
    d = 768
    
    # Try HNSW version first
    try:
        index = faiss.IndexHNSWMaxSim(d, 32)
        print("✓ Using HNSW MaxSim index")
    except:
        # Fall back to flat version
        index = faiss.IndexFlatMaxSim(d)
        print("✓ Using Flat MaxSim index")
    
    # Create test data
    n_docs = 100
    token_counts = np.random.randint(5, 20, size=n_docs).astype(np.int32)
    
    # Generate document embeddings
    doc_embeddings = []
    for tc in token_counts:
        emb = np.random.randn(tc, d).astype(np.float32)
        # Normalize for better similarity computation
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        doc_embeddings.append(emb)
    
    # Flatten for FAISS
    flat_embeddings = np.vstack(doc_embeddings).astype(np.float32)
    
    # Add to index
    index.add_with_token_counts(n_docs, flat_embeddings, token_counts)
    print(f"✓ Added {n_docs} documents to index")
    
    # Test search
    n_queries = 5
    query_token_counts = np.random.randint(3, 10, size=n_queries).astype(np.int32)
    
    query_embeddings = []
    for tc in query_token_counts:
        emb = np.random.randn(tc, d).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        query_embeddings.append(emb)
    
    flat_queries = np.vstack(query_embeddings).astype(np.float32)
    
    # Search
    k = 10
    distances, indices = index.search_with_token_counts(
        n_queries, flat_queries, query_token_counts, k
    )
    
    print("✓ Search completed")
    print(f"Top results for first query: {indices[0][:5]}")
    print(f"Distances: {distances[0][:5]}")
    
    # Verify results are sorted
    for i in range(n_queries):
        assert np.all(distances[i][:-1] >= distances[i][1:]), "Results not properly sorted"
    
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_maxsim()