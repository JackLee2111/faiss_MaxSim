// File: faiss/utils/distances_maxsim.cpp
#include <faiss/utils/distances_maxsim.h>
#include <faiss/utils/distances.h>
#include <cmath>
#include <algorithm>

namespace faiss {

float MaxSimDistance::compute_distance(
    const float* x, int n_query_tokens,
    const float* y, int n_doc_tokens,
    int d
) {
    float total_score = 0.0f;
    
    // For each query token
    for (int i = 0; i < n_query_tokens; i++) {
        const float* q_token = x + i * d;
        float max_sim = -1.0f;
        
        // Find maximum similarity with any document token
        for (int j = 0; j < n_doc_tokens; j++) {
            const float* d_token = y + j * d;
            
            // Compute cosine similarity
            float dot_product = fvec_inner_product(q_token, d_token, d);
            float q_norm = fvec_norm_L2sqr(q_token, d);
            float d_norm = fvec_norm_L2sqr(d_token, d);
            
            float similarity = dot_product / (sqrtf(q_norm * d_norm) + 1e-8f);
            max_sim = std::max(max_sim, similarity);
        }
        
        total_score += max_sim;
    }
    
    // Return negative score for min-heap compatibility
    return -total_score;
}

DistanceComputerMaxSim::DistanceComputerMaxSim(
    int d, 
    int n_query_tokens,
    const float* q,
    const int* doc_token_counts,
    const float** doc_data
) : d(d), n_query_tokens(n_query_tokens), q(q),
    doc_token_counts(doc_token_counts), doc_data(doc_data) {
    
    // Allocate and precompute query norms
    q_norms = new float[n_query_tokens];
    for (int i = 0; i < n_query_tokens; i++) {
        q_norms[i] = fvec_norm_L2sqr(q + i * d, d);
    }
}

DistanceComputerMaxSim::~DistanceComputerMaxSim() {
    delete[] q_norms;
}

void DistanceComputerMaxSim::set_query(const float* x) {
    q = x;
    // Recompute query norms
    for (int i = 0; i < n_query_tokens; i++) {
        q_norms[i] = fvec_norm_L2sqr(q + i * d, d);
    }
}

float DistanceComputerMaxSim::operator()(idx_t i) {
    int n_doc_tokens = doc_token_counts[i];
    const float* doc = doc_data[i];
    
    return MaxSimDistance::compute_distance(
        q, n_query_tokens, doc, n_doc_tokens, d
    );
}

float DistanceComputerMaxSim::symmetric_dis(idx_t i, idx_t j) {
    // For symmetric distance during graph construction
    int n_i_tokens = doc_token_counts[i];
    int n_j_tokens = doc_token_counts[j];
    const float* doc_i = doc_data[i];
    const float* doc_j = doc_data[j];
    
    // Average of both directions
    float dist_ij = MaxSimDistance::compute_distance(
        doc_i, n_i_tokens, doc_j, n_j_tokens, d
    );
    float dist_ji = MaxSimDistance::compute_distance(
        doc_j, n_j_tokens, doc_i, n_i_tokens, d
    );
    
    return (dist_ij + dist_ji) / 2.0f;
}

} // namespace faiss