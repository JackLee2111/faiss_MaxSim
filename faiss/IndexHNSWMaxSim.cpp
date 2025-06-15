// File: faiss/IndexHNSWMaxSim.cpp
#include <faiss/IndexHNSWMaxSim.h>
#include <faiss/utils/distances_maxsim.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <numeric>

namespace faiss {

IndexHNSWMaxSim::IndexHNSWMaxSim(int d, int M)
    : Index(d, METRIC_INNER_PRODUCT), M(M), 
      efConstruction(40), efSearch(16) {
    // Create proxy index for HNSW
    proxy_index = new IndexHNSWFlat(d, M, METRIC_L2);
    proxy_index->hnsw.efConstruction = efConstruction;
    proxy_index->hnsw.efSearch = efSearch;
    is_trained = true;
}

IndexHNSWMaxSim::~IndexHNSWMaxSim() {
    delete proxy_index;
}

void IndexHNSWMaxSim::add(idx_t n, const float* x) {
    FAISS_THROW_MSG("Use add_with_token_counts for MaxSim index");
}

void IndexHNSWMaxSim::add_with_token_counts(
    idx_t n, const float* x, const int* token_counts
) {
    idx_t n0 = ntotal;
    
    // Prepare proxy vectors (average embeddings for HNSW)
    std::vector<float> proxy_vectors(n * d);
    
    const float* xi = x;
    for (idx_t i = 0; i < n; i++) {
        int n_tokens = token_counts[i];
        doc_token_counts.push_back(n_tokens);
        
        // Store document embeddings
        std::vector<float> doc_emb(n_tokens * d);
        memcpy(doc_emb.data(), xi, n_tokens * d * sizeof(float));
        doc_embeddings.push_back(std::move(doc_emb));
        
        // Compute average embedding for proxy index
        float* proxy_vec = proxy_vectors.data() + i * d;
        std::fill(proxy_vec, proxy_vec + d, 0.0f);
        
        for (int j = 0; j < n_tokens; j++) {
            for (int k = 0; k < d; k++) {
                proxy_vec[k] += xi[j * d + k];
            }
        }
        
        // Normalize by number of tokens
        float norm_factor = 1.0f / n_tokens;
        for (int k = 0; k < d; k++) {
            proxy_vec[k] *= norm_factor;
        }
        
        xi += n_tokens * d;
    }
    
    // Add proxy vectors to HNSW
    proxy_index->add(n, proxy_vectors.data());
    ntotal += n;
}

void IndexHNSWMaxSim::search(
    idx_t n,
    const float* x,
    idx_t k,
    float* distances,
    idx_t* labels,
    const SearchParameters* params
) const {
    FAISS_THROW_MSG("Use search_with_token_counts for MaxSim index");
}

void IndexHNSWMaxSim::search_with_token_counts(
    idx_t n,
    const float* x,
    const int* query_token_counts,
    idx_t k,
    float* distances,
    idx_t* labels
) const {
    FAISS_THROW_IF_NOT(k > 0);
    
    if (ntotal == 0) {
        for (idx_t i = 0; i < n * k; i++) {
            distances[i] = -1e10;
            labels[i] = -1;
        }
        return;
    }
    
    // Use a larger candidate set for reranking
    idx_t k_expanded = std::min((idx_t)(k * 10), ntotal);
    
    #pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        // Get query data
        int n_query_tokens = query_token_counts[i];
        int query_offset = 0;
        for (idx_t j = 0; j < i; j++) {
            query_offset += query_token_counts[j] * d;
        }
        const float* query_data = x + query_offset;
        
        // Compute average query embedding for initial search
        std::vector<float> avg_query(d, 0.0f);
        for (int j = 0; j < n_query_tokens; j++) {
            for (int k = 0; k < d; k++) {
                avg_query[k] += query_data[j * d + k];
            }
        }
        float norm_factor = 1.0f / n_query_tokens;
        for (int k = 0; k < d; k++) {
            avg_query[k] *= norm_factor;
        }
        
        // Search with proxy index to get candidates
        std::vector<float> proxy_distances(k_expanded);
        std::vector<idx_t> proxy_labels(k_expanded);
        
        proxy_index->search(1, avg_query.data(), k_expanded, 
                           proxy_distances.data(), proxy_labels.data());
        
        // Rerank candidates using actual MaxSim distance
        std::vector<std::pair<float, idx_t>> candidates;
        candidates.reserve(k_expanded);
        
        for (idx_t j = 0; j < k_expanded; j++) {
            if (proxy_labels[j] >= 0) {
                float maxsim_dist = MaxSimDistance::compute_distance(
                    query_data, n_query_tokens,
                    doc_embeddings[proxy_labels[j]].data(), 
                    doc_token_counts[proxy_labels[j]],
                    d
                );
                candidates.push_back({maxsim_dist, proxy_labels[j]});
            }
        }
        
        // Sort by MaxSim distance (descending, since we negate in compute_distance)
        std::sort(candidates.begin(), candidates.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Copy top-k results
        float* dist_i = distances + i * k;
        idx_t* lab_i = labels + i * k;
        
        for (idx_t j = 0; j < k; j++) {
            if (j < candidates.size()) {
                dist_i[j] = candidates[j].first;
                lab_i[j] = candidates[j].second;
            } else {
                dist_i[j] = -1e10;
                lab_i[j] = -1;
            }
        }
    }
}

void IndexHNSWMaxSim::reset() {
    proxy_index->reset();
    doc_token_counts.clear();
    doc_embeddings.clear();
    ntotal = 0;
}

} // namespace faiss