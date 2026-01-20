#pragma once

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/detail/ygm_ptr.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <iostream>


template <typename Key, typename Value>
class proc_cache{
    static_assert(std::is_trivially_copyable_v<Key>);
    static_assert(std::is_trivially_copyable_v<Value>);
    static constexpr size_t NUM_ENTRIES = 1000000;

public:
    using internal_container_type = ygm::container::map<Key, Value>;
    using key_type = Key;
    using value_type = Value;


    /**
     * @brief constructor for processor-local cache
     * 
     * @param local_nzz : number of nonzero entries in an unsorted matrix that a caller rank owns
     * @param total_unique_rows : total number of unique nonzero rows in a sorted matrix
     * @param total_nnz : total number of nonzero entries in a sorted matrix
     */
    explicit proc_cache(ygm::comm &c, internal_container_type &accum,
                        int64_t total_nnz, int64_t total_unique_rows, int32_t local_entries) : m_comm(c), m_map(accum)
        {

        // obtain the number of nonzero rows and the number of rows in the sorted matrix 

        // average number of nonzero entries per row
        unsigned long avg_deg = total_nnz / total_unique_rows;

        // if a rank owns 10 entries, then each entry will multiply with the entire matching row
        // so in average, how many multiplications will a rank perform?
        unsigned long estimated_entries = local_entries * avg_deg;
        double deduplication_factor = 0.3; 
        // estimated number of distinct number of multiplication pairs (i, j)
        unsigned long estimated_size = estimated_entries * deduplication_factor;
        m_cache.reserve(NUM_ENTRIES);

    }

    void cache_insert(const key_type &key, const value_type &value){
        if (m_cache_empty) {
            m_cache_empty = false;
            // m_map.comm().register_pre_barrier_callback(
            //     [this]() { this->cache_flush_all(); });
        }

        auto it = m_cache.find(key);
        if(it != m_cache.end()){
            m_cache.at(key) += value;
            local_accumulate++;
        }
        else{
            m_cache.insert({key, value});
        }

        if(m_cache.size() >= NUM_ENTRIES){
            cache_flush_all();
        }
    }

     /**
     * @brief Flushes a slot in the cache to the map
     * 
     * @param entry: which specific entry it should flush
     */
    void cache_flush(const key_type &key){
        auto it = m_cache.find(key);
        if(it != m_cache.end()){
            auto cached_value = it->second;
            m_map.async_visit(
                key,
                [](const key_type &key, value_type &partial_product, value_type to_add){
                    partial_product += to_add;
                },
                cached_value
            );
            local_flush++;
        }
    }   

    
    void cache_flush_all() {
        if (!m_cache_empty) {
            for (auto &p : m_cache) {
                cache_flush(p.first);
            }
            m_cache.clear();
            //m_cache.rehash(0);
            m_cache_empty = true;
            global_flush++;
        }
    }

    void cache_print(){
        m_comm.cout("local accumulate: ", local_accumulate, 
                        ", local flush: ", local_flush, 
                        ", global flush: ", global_flush);
    }


private: 
    ygm::comm                                    &m_comm;
    boost::unordered_flat_map<Key, int64_t>         m_cache;
    bool                                         m_cache_empty = true;
    internal_container_type                      &m_map;
    int                                          local_accumulate = 0;
    int                                          local_flush = 0;
    int                                          global_flush = 0;
};