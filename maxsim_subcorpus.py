# maxsim_subcorpus.py
import numpy as np
import torch
import pickle
import time
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import faiss

class MaxSimSubCorpusIndexer:
    def __init__(self, d=768, M=32, ef_construction=40, ef_search=16):
        """Initialize MaxSim indexer with sub-corpus support"""
        self.d = d
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Initialize model
        self.sbert_path = "sentence-transformers/msmarco-roberta-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.sbert_path)
        self.model = AutoModel.from_pretrained(self.sbert_path)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def encode_token_embeddings(self, texts, batch_size=32):
        """Encode texts into token embeddings with progress bar"""
        embeddings = []
        token_counts = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i+batch_size]
            
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
            
            for j, text in enumerate(batch_texts):
                mask = inputs['attention_mask'][j]
                valid_tokens = mask.sum().item()
                token_emb = outputs[j, 1:valid_tokens-1].cpu()
                
                embeddings.append(token_emb)
                token_counts.append(token_emb.shape[0])
        
        return embeddings, token_counts
    
    def build_subcorpus_indices(self, all_subcorpus_embeddings, cache_dir="index_cache"):
        """
        Build indices for multiple sub-corpora
        
        Args:
            all_subcorpus_embeddings: List of lists, each containing embeddings for a sub-corpus
            cache_dir: Directory to cache indices
            
        Returns:
            indices: List of built indices
            build_times: List of build times
        """
        os.makedirs(cache_dir, exist_ok=True)
        indices = []
        build_times = []
        
        for i, subcorpus_emb in enumerate(all_subcorpus_embeddings):
            index_path = os.path.join(cache_dir, f"subcorpus_{i}.idx")
            
            # Check if index exists
            if os.path.exists(index_path):
                print(f"Loading cached index for sub-corpus {i}")
                with open(index_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    indices.append(saved_data['index'])
                    build_times.append(saved_data['build_time'])
            else:
                print(f"\nBuilding index for sub-corpus {i} ({len(subcorpus_emb)} documents)")
                
                # Get token counts
                token_counts = [emb.shape[0] for emb in subcorpus_emb]
                
                # Build index
                start_time = time.time()
                
                index = faiss.IndexHNSWMaxSim(self.d, self.M)
                index.efConstruction = self.ef_construction
                index.efSearch = self.ef_search
                
                # Flatten embeddings
                flattened_embeddings = []
                for emb in subcorpus_emb:
                    flattened_embeddings.extend(emb.numpy().flatten())
                
                flattened_embeddings = np.array(flattened_embeddings, dtype=np.float32)
                token_counts_array = np.array(token_counts, dtype=np.int32)
                
                # Add to index
                n_docs = len(subcorpus_emb)
                index.add_with_ids_and_tokens(
                    n_docs,
                    flattened_embeddings,
                    np.arange(n_docs, dtype=np.int64),
                    token_counts_array
                )
                
                build_time = time.time() - start_time
                build_times.append(build_time)
                indices.append(index)
                
                print(f"Sub-corpus {i} index built in {build_time:.2f} seconds")
                
                # Save index
                with open(index_path, 'wb') as f:
                    pickle.dump({
                        'index': index,
                        'token_counts': token_counts,
                        'build_time': build_time
                    }, f)
        
        return indices, build_times
    
    def search_all_subcorpora(self, indices, query_embeddings, query_token_counts, k=10):
        """
        Search across all sub-corpus indices
        
        Args:
            indices: List of FAISS indices
            query_embeddings: Query embeddings
            query_token_counts: Query token counts
            k: Number of results per sub-corpus
            
        Returns:
            all_results: Results from all sub-corpora
            search_time: Total search time
        """
        print(f"\nSearching {len(indices)} sub-corpus indices...")
        start_time = time.time()
        
        all_results = []
        
        for i, index in enumerate(indices):
            # Flatten query embeddings
            flattened_queries = []
            for emb in query_embeddings:
                flattened_queries.extend(emb.numpy().flatten())
            
            flattened_queries = np.array(flattened_queries, dtype=np.float32)
            query_token_counts_array = np.array(query_token_counts, dtype=np.int32)
            
            # Search this sub-corpus
            distances, indices_result = index.search_with_tokens(
                len(query_embeddings),
                flattened_queries,
                query_token_counts_array,
                k
            )
            
            distances = distances.reshape(len(query_embeddings), k)
            indices_result = indices_result.reshape(len(query_embeddings), k)
            
            all_results.append({
                'subcorpus_id': i,
                'distances': distances,
                'indices': indices_result
            })
        
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.4f} seconds")
        
        return all_results, search_time
    
    def merge_results(self, all_results, query_token_counts, k=10):
        """
        Merge results from all sub-corpora and return top-k
        
        Args:
            all_results: Results from all sub-corpora
            query_token_counts: Token counts for queries
            k: Number of final results
            
        Returns:
            merged_results: Top-k results across all sub-corpora
        """
        n_queries = len(query_token_counts)
        merged_results = []
        
        for q_idx in range(n_queries):
            # Collect all results for this query
            all_distances = []
            all_indices = []
            all_subcorpus_ids = []
            
            for result in all_results:
                distances = result['distances'][q_idx]
                indices = result['indices'][q_idx]
                subcorpus_id = result['subcorpus_id']
                
                all_distances.extend(distances)
                all_indices.extend(indices)
                all_subcorpus_ids.extend([subcorpus_id] * len(distances))
            
            # Sort by distance (ascending, since we use negative similarity)
            sorted_idx = np.argsort(all_distances)[:k]
            
            merged_results.append({
                'distances': [all_distances[i] for i in sorted_idx],
                'indices': [all_indices[i] for i in sorted_idx],
                'subcorpus_ids': [all_subcorpus_ids[i] for i in sorted_idx]
            })
        
        return merged_results

def demo_subcorpus_indexing():
    """Demo with multiple sub-corpora"""
    
    # Create multiple sub-corpora
    subcorpus1 = [
        "Wine pairs well with cheese and steak.",
        "The best wines come from specific regions.",
        "French cuisine often features wine pairings.",
        "Wine regions produce different grape varieties.",
        "Red wine complements red meat dishes."
    ]
    
    subcorpus2 = [
        "Rice and tea are staples in Chinese meals.",
        "Tea ceremonies are important in Asian culture.",
        "Chinese tea has many health benefits.",
        "Green tea is popular in Japan.",
        "Traditional tea preparation requires skill."
    ]
    
    subcorpus3 = [
        "People enjoy wine tasting in France.",
        "Steak and wine make a classic dinner combination.",
        "Traditional meals vary by culture and region.",
        "Food and beverage pairings enhance dining.",
        "Cultural dining experiences differ worldwide."
    ]
    
    all_subcorpora = [subcorpus1, subcorpus2, subcorpus3]
    
    queries = [
        "What food goes well with wine?",
        "Tell me about tea in Asian culture",
        "Wine and food pairings"
    ]
    
    # Initialize indexer
    indexer = MaxSimSubCorpusIndexer(d=768, M=32)
    
    # Encode all sub-corpora
    all_subcorpus_embeddings = []
    print("Encoding sub-corpora...")
    
    for i, subcorpus in enumerate(all_subcorpora):
        print(f"\nEncoding sub-corpus {i}...")
        emb_list, token_counts = indexer.encode_token_embeddings(subcorpus)
        all_subcorpus_embeddings.append(emb_list)
    
    # Build indices
    indices, build_times = indexer.build_subcorpus_indices(all_subcorpus_embeddings)
    
    # Encode queries
    print("\nEncoding queries...")
    query_emb_list, query_token_counts = indexer.encode_token_embeddings(queries)
    
    # Search all sub-corpora
    all_results, search_time = indexer.search_all_subcorpora(
        indices, query_emb_list, query_token_counts, k=3
    )
    
    # Merge results
    merged_results = indexer.merge_results(all_results, query_token_counts, k=5)
    
    # Display results
    print("\nMerged Search Results:")
    print("=" * 80)
    
    for i, (query, result) in enumerate(zip(queries, merged_results)):
        print(f"\nQuery {i+1}: {query}")
        print("-" * 40)
        
        for rank, (dist, idx, subcorpus_id) in enumerate(
            zip(result['distances'], result['indices'], result['subcorpus_ids'])
        ):
            similarity = -dist / query_token_counts[i]
            doc_text = all_subcorpora[subcorpus_id][idx]
            print(f"Rank {rank+1}: Sub-corpus {subcorpus_id}, Doc {idx} (Similarity: {similarity:.4f})")
            print(f"   Text: {doc_text}")
            print()
    
    # Performance summary
    print("\nPerformance Summary:")
    print(f"Total index construction time: {sum(build_times):.2f} seconds")
    print(f"Average per sub-corpus: {np.mean(build_times):.2f} seconds")
    print(f"Total search time: {search_time:.4f} seconds")
    print(f"Average time per query: {search_time/len(queries)*1000:.2f} ms")

if __name__ == "__main__":
    demo_subcorpus_indexing()