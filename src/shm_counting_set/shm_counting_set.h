#pragma once

#include <ygm/comm.hpp>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <ygm/detail/layout.hpp>
#include <ygm/container/map.hpp>
#include <sys/mman.h>   // For shm_open, mmap
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <pthread.h>
#include <ygm/container/detail/base_misc.hpp>


#define SIZE 1024 * 1024    // million entries
#define SHM_SIZE  (SIZE * sizeof(Entry)) // total size in bytes
/*
    use static assert to explicitly define the allowed datatype

    shm only allows trivially copyable datatype (can be copied bit by bit); It does
    not allow datatype of vector, string, etc. because they use heap memory to store data 
    (heap pointer is local to the process. If a different process tries to use another process'
    heap pointer, it will lead to seg fault).

*/
// explicitly define the datatype
// plain old data, trivially copyable
// static_assert
template <typename Key, typename Value>
class shm_counting_set{

public:
    using self_type = shm_counting_set<Key, Value>;
    using internal_container_type = ygm::container::map<Key, Value>;
    using key_type = Key;
    using value_type = Value;

    explicit shm_counting_set(ygm::comm &c, internal_container_type &accum) : 
                            m_comm(c), 
                            m_local_size(m_comm.layout().local_size()),
                            m_local_id(m_comm.layout().local_id()),
                            m_node_id(m_comm.layout().node_id()),
                            m_map(accum), 
                            m_bip_ptrs(m_local_size){

        // printf("Rank %d, node size: %d, local id: %d, node id: %d\n", 
        //         m_comm.rank(), 
        //         m_local_size,
        //         m_local_id,
        //         m_node_id);
        /*
            Create shared memory files
        */
        std::string filename_s = "/BIP_" + std::to_string(m_comm.rank());
        const char *filename_c = filename_s.c_str();

        int fd = shm_open(filename_c, O_CREAT | O_RDWR, 0666);
        if(fd == -1){
            perror("shm_open() failed\n");
            return;
        }

        if(ftruncate(fd, SHM_SIZE) == -1){
            perror("ftruncate() failed\n");
            return;
        }
        // fd is no longer needed
        close(fd);

        std::vector<std::string> BIP_filenames(m_local_size);
        // merge filenames to the local id 0 rank within the same physical node
        /*
            does not require pthis (distributed ygm pointer) because local rank 0 is communicating to processors 
            within the same node, eliminating the need to serialize the lambda.
            *raw pointer cannot be serialized, only ygm pointer can.
        */
        auto collect_filenames = [this, &BIP_filenames](int local_id, int node_id, int global_rank, std::string filename){
            //printf("Rank %d, Received filename %s from local rank %d, global rank %d\n", m_comm.rank(), filename.c_str(), local_id, global_rank);
            BIP_filenames.at(local_id) = filename;
        };
        
        int local_rank_zero = m_comm.rank() - m_local_id;
        m_comm.async(local_rank_zero, collect_filenames, m_local_id, m_node_id, m_comm.rank(), filename_s);
        m_comm.barrier();

        // broadcast the filenames (within the same node)
        auto broadcastBIP = [&BIP_filenames](std::vector<std::string> incoming_filenames){
            BIP_filenames = incoming_filenames;
        };
        if(m_comm.rank() == local_rank_zero){
            for(int i = local_rank_zero + 1; i < local_rank_zero + m_local_size; i++){
                //m_comm.cout("Sharing filenames to rank ", i);
                m_comm.async(i, broadcastBIP, BIP_filenames);
            }
        }
        m_comm.barrier();

        for(int i = 0; i < BIP_filenames.size(); i++){
            filename_to_BIP(m_bip_ptrs, i, BIP_filenames[i], SHM_SIZE);
        }

         // some processes may unlink before others get the chance to shm_open
        m_comm.barrier();
        shm_unlink(filename_c);

        // initialize the shared memory file with the struct data
        /*
            Could there be a better method to avoid this pre-processing? Or is this so computationally trivial that
            it does not matter?
        */
        Entry default_entry{};      

        // DO NOT OVERWRITE THE FIRST ELEMENT. IT CONTAINS THE MUTEX OF THE SHARED MEMORY FILE
        for (size_t i = 1; i < SIZE; i++) {
            Entry* entry = (Entry*)m_bip_ptrs[m_local_id].get();
            std::memcpy(entry + i,
                        &default_entry,
                        sizeof(Entry));
        }
    }

    ~shm_counting_set() {
        m_comm.barrier();
        //m_comm.log(log_level::info, "Destroying shm_counting_set");
    }

    void cache_insert(const key_type &key, const value_type &value, int &flush_count){
        if(m_cache_empty){
            m_cache_empty = false;
            m_map.comm().register_pre_barrier_callback(
                    [this]() { 
                        this->value_cache_flush_all(); 
                    }
                );
        }
        int BIP_index = ygm::container::detail::hash<key_type>{}(key) % m_local_size;
        int slot = ygm::container::detail::hash<key_type>{}(key) % SIZE;

        Entry* shared_region = (Entry*)m_bip_ptrs[BIP_index].get();
        Entry* cached_entry = shared_region + slot;

        //m_comm.cout(m_comm.rank(), " is locking local region ", BIP_index);
        pthread_mutex_lock(&(shared_region->mutex));
        if(cached_entry->s_value == -1){ // why did I think that it should have been (!=)?
            //m_comm.cout("New entry: Assigning ", value);
            cached_entry->s_key = key;
            cached_entry->s_value = value;
        }
        else{
            YGM_ASSERT_DEBUG(cached_entry->s_value > 0);
            // if the key matches
            if(cached_entry->s_key == key){
                //m_comm.cout("key matched. Adding ", value);
                cached_entry->s_value += value;
            }
            else{ // different key
                // flush the slot
                value_cache_flush(BIP_index, slot);
                flush_count++;
                cached_entry->s_key = key;
                cached_entry->s_value = value;
            }
        }
        // if the cached value is greater than the value of int32, then flush that slot
        if(cached_entry->s_value >= std::numeric_limits<int32_t>::max() / 2){
            value_cache_flush(BIP_index, slot);
            flush_count++;
        }
        //m_comm.cout(m_comm.rank(), " is unlocking local region ", BIP_index);
        pthread_mutex_unlock(&(shared_region->mutex));
    }

    /**
     * @brief Flushes a slot in the cache to the map
     * 
     * @param BIP_index: which shared memory file it should flush in
     * @param slot: which slot to flush
     */
    void value_cache_flush(int BIP_index, size_t slot){
        YGM_ASSERT_DEBUG(BIP_index < m_local_size);
        YGM_ASSERT_DEBUG(slot < SIZE);

        Entry* entry = (Entry*)m_bip_ptrs[BIP_index].get();
        auto key          = (entry + slot)->s_key;
        auto cached_value = (entry + slot)->s_value;
        YGM_ASSERT_DEBUG(cached_value > 0);

        // call async visit
        m_map.async_visit(
            key,
            [](const key_type &key, value_type &partial_product, value_type to_add){
                partial_product += to_add;
            },
            cached_value
        );
        (entry + slot)->s_key = key_type();
        (entry + slot)->s_value = -1;
    }

    /**
     * @brief Each processor flushes its own shared memory region
     */
    void value_cache_flush_all(){
        if(!m_cache_empty){
            Entry* entry = (Entry*)m_bip_ptrs[m_local_id].get();
            for(int i = 0; i < SIZE; i++){
                if((entry + i)->s_value > 0){
                    value_cache_flush(m_local_id, i);
                }
            }
            m_cache_empty = false;
        }
    }

private:

    struct Entry{
        Key s_key = key_type();
        Value s_value = -1;
        pthread_mutex_t mutex;
    };

    struct MMapDestructor{
        size_t size;

        MMapDestructor(size_t s = 0) : size(s){} // constructor

        // when unique pointer goes out of scope, it calls the 
        // deleter function
        void operator()(void* ptr) const{
            if(ptr != MAP_FAILED){
                munmap(ptr, size);
            }
        }
    };

    void filename_to_BIP(std::vector<std::unique_ptr<void, MMapDestructor>> &ptrs_vec, int local_id, std::string filename, int size){  
        // open a file descriptor to the shared file 
        int fd = shm_open(filename.c_str(), O_RDWR, 0666);
        if (fd == -1) {
            perror("Opening received shm failed");
            return;
        }

        void *mmap_ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, 
                        MAP_SHARED, fd, 0);
        if(mmap_ptr == MAP_FAILED){
            perror("mapping to *shared* BIP failed\n");
            return;
        }
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&((Entry*)mmap_ptr)->mutex, &attr);
        pthread_mutexattr_destroy(&attr);

        close(fd);

        ((Entry*)mmap_ptr)->s_key = key_type();
        ((Entry*)mmap_ptr)->s_value = -1;
        ptrs_vec.at(local_id) = std::unique_ptr<void, MMapDestructor>(mmap_ptr, MMapDestructor(size));
    }


    // m_ is a naming convention to indicate "member"
    ygm::comm                                          &m_comm;
    // physical node size
    int                                                m_local_size = -1;
    // a vector of BIP pointers
    std::vector<std::unique_ptr<void, MMapDestructor>> m_bip_ptrs;
    int                                                m_local_id = -1;
    int                                                m_node_id = -1;

    bool                                               m_cache_empty = true;
    internal_container_type                            &m_map;
};


#include "shm_counting_set.ipp"