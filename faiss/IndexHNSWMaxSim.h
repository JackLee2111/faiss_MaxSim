// File: faiss/IndexHNSWMaxSim.h
#ifndef FAISS_INDEX_HNSW_MAXSIM_H
#define FAISS_INDEX_HNSW_MAXSIM_H

#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances_maxsim.h>
#include <vector>

namespace faiss {

/**
 * MaxSim index using HNSW for approximate search.
 * This implementation stores document embeddings and uses a proxy index
 * for HNSW graph construction.
 */
struct IndexHNSWMaxSim : Index {
    // Storage for multi-token documents
    std::vector<int> doc_token_counts;
    std::vector<std::vector<float>> doc_embeddings;
    
    // Proxy index for HNSW - stores average embeddings
    IndexHNSWFlat* proxy_index;
    
    // Parameters
    int M;
    int efConstruction;
    int efSearch;
    
    explicit IndexHNSWMaxSim(int d, int M = 32);
    ~IndexHNSWMaxSim() override;
    
    void add(idx_t n, const float* x) override;
    void add_with_token_counts(idx_t n, const float* x, const int* token_counts);
    
    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params = nullptr) const override;
    
    void search_with_token_counts(
        idx_t n,
        const float* x,
        const int* query_token_counts,
        idx_t k,
        float* distances,
        idx_t* labels) const;
    
    void reset() override;
    void train(idx_t n, const float* x) override { is_trained = true; }
    
    void reconstruct(idx_t key, float* recons) const override {
        FAISS_THROW_MSG("reconstruct not implemented for IndexHNSWMaxSim");
    }
};

} // namespace faiss

#endif