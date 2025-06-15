# maxsim_faiss_demo.py
import numpy as np
import torch
import pickle
import time
import os
from transformers import AutoTokenizer, AutoModel
import faiss

class MaxSimFAISSIndex:
    def __init__(self, d=768, M=32, ef_construction=40, ef_search=16):
        """
        Initialize MaxSim FAISS Index
        
        Args:
            d: Embedding dimension
            M: Number of connections in HNSW graph
            ef_construction: Size of dynamic candidate list
            ef_search: Size of dynamic candidate list for search
        """
        self.d = d
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Initialize model and tokenizer
        self.sbert_path = "sentence-transformers/msmarco-roberta-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.sbert_path)
        self.model = AutoModel.from_pretrained(self.sbert_path)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def encode_token_embeddings(self, texts, batch_size=32):
        """
        Encode texts into token-level embeddings
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            embeddings: List of tensors, each of shape [n_tokens, d]
            token_counts: List of token counts for each text
        """
        embeddings = []
        token_counts = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and encode
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs).last_hidden_state
            
            # Process each text in batch
            for j, text in enumerate(batch_texts):
                # Get attention mask to find actual tokens
                mask = inputs['attention_mask'][j]
                valid_tokens = mask.sum().item()
                
                # Extract embeddings (remove [CLS] and [SEP])
                token_emb = outputs[j, 1:valid_tokens-1].cpu()
                
                embeddings.append(token_emb)
                token_counts.append(token_emb.shape[0])
        
        return embeddings, token_counts
    
    def pad_embeddings(self, embeddings, max_tokens=None):
        """
        Pad embeddings to have consistent token counts
        
        Args:
            embeddings: List of tensors with varying token counts
            max_tokens: Maximum number of tokens (if None, use max from data)
            
        Returns:
            padded_embeddings: Numpy array of shape [n_docs, max_tokens, d]
            token_counts: Original token counts
        """
        token_counts = [emb.shape[0] for emb in embeddings]
        
        if max_tokens is None:
            max_tokens = max(token_counts)
        
        n_docs = len(embeddings)
        padded_embeddings = np.zeros((n_docs, max_tokens, self.d), dtype=np.float32)
        
        for i, emb in enumerate(embeddings):
            n_tokens = min(emb.shape[0], max_tokens)
            padded_embeddings[i, :n_tokens] = emb[:n_tokens].numpy()
        
        return padded_embeddings, token_counts
    
    def build_index(self, corpus_embeddings, token_counts, index_path=None):
        """
        Build HNSW index with MaxSim distance
        
        Args:
            corpus_embeddings: List of document embeddings
            token_counts: List of token counts for each document
            index_path: Path to save the index
            
        Returns:
            index: Built FAISS index
            build_time: Time taken to build index
        """
        print("Building MaxSim HNSW index...")
        start_time = time.time()
        
        # Create MaxSim index
        index = faiss.IndexHNSWMaxSim(self.d, self.M)
        index.efConstruction = self.ef_construction
        index.efSearch = self.ef_search
        
        # Flatten embeddings for FAISS
        flattened_embeddings = []
        for emb in corpus_embeddings:
            flattened_embeddings.extend(emb.numpy().flatten())
        
        flattened_embeddings = np.array(flattened_embeddings, dtype=np.float32)
        token_counts_array = np.array(token_counts, dtype=np.int32)
        
        # Add to index
        n_docs = len(corpus_embeddings)
        index.add_with_ids_and_tokens(
            n_docs,
            flattened_embeddings,
            np.arange(n_docs, dtype=np.int64),
            token_counts_array
        )
        
        build_time = time.time() - start_time
        print(f"Index built in {build_time:.2f} seconds")
        
        # Save index if path provided
        if index_path:
            with open(index_path, 'wb') as f:
                pickle.dump({
                    'index': index,
                    'token_counts': token_counts,
                    'build_time': build_time
                }, f)
            print(f"Index saved to {index_path}")
        
        return index, build_time
    
    def search(self, index, query_embeddings, query_token_counts, k=10):
        """
        Search the index using MaxSim similarity
        
        Args:
            index: FAISS MaxSim index
            query_embeddings: List of query embeddings
            query_token_counts: List of query token counts
            k: Number of results to retrieve
            
        Returns:
            results: List of (distances, indices) for each query
            search_time: Time taken for search
        """
        print(f"Searching for {len(query_embeddings)} queries...")
        start_time = time.time()
        
        # Flatten query embeddings
        flattened_queries = []
        for emb in query_embeddings:
            flattened_queries.extend(emb.numpy().flatten())
        
        flattened_queries = np.array(flattened_queries, dtype=np.float32)
        query_token_counts_array = np.array(query_token_counts, dtype=np.int32)
        
        # Search
        distances, indices = index.search_with_tokens(
            len(query_embeddings),
            flattened_queries,
            query_token_counts_array,
            k
        )
        
        # Reshape results
        distances = distances.reshape(len(query_embeddings), k)
        indices = indices.reshape(len(query_embeddings), k)
        
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.4f} seconds")
        
        return list(zip(distances, indices)), search_time

def main():
    # Initialize index
    indexer = MaxSimFAISSIndex(d=768, M=32, ef_construction=40, ef_search=16)
    
    # Example corpus and queries
    corpus = [
        "Wine pairs well with cheese and steak.",
        "Rice and tea are staples in Chinese meals.",
        "People enjoy wine tasting in France.",
        "The best wines come from specific regions.",
        "Tea ceremonies are important in Asian culture.",
        "Steak and wine make a classic dinner combination.",
        "French cuisine often features wine pairings.",
        "Chinese tea has many health benefits.",
        "Wine regions produce different grape varieties.",
        "Traditional meals vary by culture and region."
    ]
    
    queries = [
        "What food goes well with wine?",
        "Tell me about tea in Asian culture",
        "Wine and food pairings"
    ]
    
    # Encode corpus
    print("Encoding corpus...")
    corpus_emb_list, corpus_token_counts = indexer.encode_token_embeddings(corpus)
    print(f"Encoded {len(corpus)} documents")
    
    # Build or load index
    index_path = "maxsim_index.pkl"
    
    if os.path.exists(index_path):
        print(f"Loading existing index from {index_path}")
        with open(index_path, 'rb') as f:
            saved_data = pickle.load(f)
            index = saved_data['index']
            build_time = saved_data['build_time']
        print(f"Index loaded (originally built in {build_time:.2f} seconds)")
    else:
        index, build_time = indexer.build_index(
            corpus_emb_list, 
            corpus_token_counts, 
            index_path
        )
    
    # Encode queries
    print("\nEncoding queries...")
    query_emb_list, query_token_counts = indexer.encode_token_embeddings(queries)
    
    # Search
    results, search_time = indexer.search(
        index, 
        query_emb_list, 
        query_token_counts, 
        k=5
    )
    
    # Display results
    print("\nSearch Results:")
    print("=" * 80)
    
    for i, (query, (distances, indices)) in enumerate(zip(queries, results)):
        print(f"\nQuery {i+1}: {query}")
        print("-" * 40)
        
        for rank, (dist, idx) in enumerate(zip(distances, indices)):
            # Convert negative distance back to similarity score
            similarity = -dist / query_token_counts[i]
            print(f"Rank {rank+1}: Doc {idx} (Similarity: {similarity:.4f})")
            print(f"   Text: {corpus[idx]}")
            print()
    
    # Performance summary
    print("\nPerformance Summary:")
    print(f"Index construction time: {build_time:.2f} seconds")
    print(f"Search time: {search_time:.4f} seconds")
    print(f"Average time per query: {search_time/len(queries)*1000:.2f} ms")
    
    # Save results
    results_data = {
        'corpus': corpus,
        'queries': queries,
        'results': results,
        'corpus_embeddings': corpus_emb_list,
        'query_embeddings': query_emb_list,
        'build_time': build_time,
        'search_time': search_time
    }
    
    with open('search_results.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    print("\nResults saved to search_results.pkl")

if __name__ == "__main__":
    main()