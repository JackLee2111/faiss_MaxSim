// File: faiss/utils/distances_maxsim.h
#ifndef FAISS_DISTANCES_MAXSIM_H
#define FAISS_DISTANCES_MAXSIM_H

#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/Index.h>

namespace faiss {

// MaxSim distance computation for multi-vector embeddings
struct MaxSimDistance {
    // Compute MaxSim distance between two multi-vector embeddings
    // x: query embeddings [n_query_tokens x d]
    // y: document embeddings [n_doc_tokens x d]
    // Returns negative MaxSim score (for min-heap compatibility)
    static float compute_distance(
        const float* x, int n_query_tokens,
        const float* y, int n_doc_tokens,
        int d
    );
};

// Distance computer for MaxSim
struct DistanceComputerMaxSim : DistanceComputer {
    int d;                    // dimension
    int n_query_tokens;       // number of query tokens
    const float* q;           // query data
    float* q_norms;           // precomputed query norms (allocated)
    
    // Document metadata storage
    const int* doc_token_counts;  // number of tokens per document
    const float** doc_data;       // pointers to document embeddings
    
    DistanceComputerMaxSim(
        int d, 
        int n_query_tokens,
        const float* q,
        const int* doc_token_counts,
        const float** doc_data
    );
    
    virtual ~DistanceComputerMaxSim();
    
    void set_query(const float* x) override;
    float operator()(idx_t i) override;
    float symmetric_dis(idx_t i, idx_t j) override;
};

} // namespace faiss

#endif