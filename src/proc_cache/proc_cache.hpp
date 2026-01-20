#pragma once

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/detail/ygm_ptr.hpp>
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
    explicit proc_cache(ygm::comm &c, internal_container_type &accum) : m_comm(c), m_map(accum)
        {

        // obtain the number of nonzero rows and the number of rows in the sorted matrix 

        // average number of nonzero entries per row
        // unsigned long avg_deg = total_nnz / total_unique_rows;

        // // if a rank owns 10 entries, then each entry will multiply with the entire matching row
        // // so in average, how many multiplications will a rank perform?
        // unsigned long estimated_entries = local_entries * avg_deg;
        // double deduplication_factor = 0.5; 
        // // estimated number of distinct number of multiplication pairs (i, j)
        // unsigned long estimated_size = estimated_entries * deduplication_factor;

        m_cache.resize(NUM_ENTRIES, {key_type(), -1});

    }

    void cache_insert(const key_type &key, const value_type &value){
        if (m_cache_empty) {
            m_cache_empty = false;
            // m_map.comm().register_pre_barrier_callback(
            //     [this]() { this->cache_flush_all(); });
        }
        int slot = ygm::container::detail::hash<key_type>{}(key) % NUM_ENTRIES;

        if (m_cache[slot].second == -1) {
            m_cache[slot].first  = key;
            m_cache[slot].second = value;
        } 
        else {
            YGM_ASSERT_DEBUG(m_cache[slot].second > 0);
            if (m_cache[slot].first == key) {
                local_accumulate++;
                m_cache[slot].second += value;
            } else {
                cache_flush(slot);
                YGM_ASSERT_DEBUG(m_cache[slot].second == -1);
                m_cache[slot].first  = key;
                m_cache[slot].second = value;
            }
        }
        // if the cached value is greater than the value of int64, then flush that slot
        // need to consider cases in which values are negative, but assume positive values for now.
        if(m_cache[slot].second >= std::numeric_limits<int64_t>::max() / 2){
            cache_flush(slot);
        }
    }

     /**
     * @brief Flushes a slot in the cache to the map
     * 
     * @param entry: which specific entry it should flush
     */
    void cache_flush(size_t slot){
        auto key          = m_cache[slot].first;
        auto cached_value = m_cache[slot].second;
        YGM_ASSERT_DEBUG(cached_value > 0);
        m_map.async_visit(
            key,
            [](const key_type &key, value_type &partial_product, value_type to_add){
                partial_product += to_add;
            },
            cached_value
        );
        m_cache[slot].first  = key_type();
        m_cache[slot].second = -1;
        local_flush++;
    }   

    
    void cache_flush_all() {
        if (!m_cache_empty) {
            for (size_t i = 0; i < m_cache.size(); i++) {
                if (m_cache[i].second > 0) {
                    cache_flush(i);
                    global_flush++;
                }
            }
            m_cache_empty = true;
            m_comm.cout("local accumulate: ", local_accumulate, 
                        ", local flush: ", local_flush, 
                        ", global flush: ", global_flush);
        }
    }


private: 
    ygm::comm                                    &m_comm;
    std::vector<std::pair<Key, int64_t>>         m_cache;
    bool                                         m_cache_empty = true;
    internal_container_type                      &m_map;
    int                                          local_accumulate = 0;
    int                                          local_flush = 0;
    int                                          global_flush = 0;
};