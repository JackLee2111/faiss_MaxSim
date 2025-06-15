import numpy as np
import faiss

class IndexHNSWMaxSimWrapper:
    """Python wrapper for IndexHNSWMaxSim that handles numpy conversions"""
    
    def __init__(self, d, M=32):
        # Access the C++ class directly
        self.index = faiss.IndexHNSWMaxSim(d, M)
        self.d = d
        
    def add_with_token_counts(self, embeddings, token_counts):
        """Add documents with variable token counts
        
        Args:
            embeddings: numpy array of all token embeddings concatenated
            token_counts: numpy array of token counts per document
        """
        n = len(token_counts)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        token_counts = np.ascontiguousarray(token_counts, dtype=np.int32)
        
        # Call the C++ method directly
        self.index.add_with_token_counts(n, 
            faiss.swig_ptr(embeddings), 
            faiss.swig_ptr(token_counts))
    
    def search_with_token_counts(self, queries, query_token_counts, k):
        """Search with variable token queries
        
        Args:
            queries: numpy array of all query token embeddings concatenated
            query_token_counts: numpy array of token counts per query
            k: number of results per query
            
        Returns:
            distances, indices: numpy arrays of shape (n_queries, k)
        """
        n = len(query_token_counts)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        query_token_counts = np.ascontiguousarray(query_token_counts, dtype=np.int32)
        
        # Allocate output arrays
        distances = np.empty((n, k), dtype=np.float32)
        indices = np.empty((n, k), dtype=np.int64)
        
        # Call the C++ method
        self.index.search_with_token_counts(n,
            faiss.swig_ptr(queries),
            faiss.swig_ptr(query_token_counts),
            k,
            faiss.swig_ptr(distances),
            faiss.swig_ptr(indices))
        
        return distances, indices
    
    @property
    def ntotal(self):
        return self.index.ntotal
    
    @property
    def is_trained(self):
        return self.index.is_trained