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
#include <atomic>
#include <ygm/container/detail/base_misc.hpp>
#include <unistd.h>
#include <stdio.h>  


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
class shm_cache{
    static_assert(std::is_trivially_copyable_v<Key>);
    static_assert(std::is_trivially_copyable_v<Value>);

private:

    static constexpr size_t NUM_ENTRIES = 1024 * 1024;
    /*
        Key s_key = key_type();
        Value s_value = -1;
        implicitly creates a constructor which is not trivially copyable.
        mmap gives a raw memory address and creating Entry object in the shared memory
        means that the default constructor is not called.
    */
    struct Entry{
        Key s_key;
        Value s_value;
    };

    /*
        this somehow prevents entry writing from overwriting part of the pthread mutex?
    */
    struct shm_layout{
        // 40 bytes big
        // alignment of 8 bytes
        Entry entries[NUM_ENTRIES];
    };

    // custom string object to avoid heap pointers
    struct m_string{
        char data[256];

        template <class Archive>
        void serialize(Archive& ar) {
            ar(data);
        }
    };

    static constexpr size_t SHM_SIZE = sizeof(shm_layout);
    
public:
    using self_type = shm_cache<Key, Value>;
    using internal_container_type = ygm::container::map<Key, Value>;
    using key_type = Key;
    using value_type = Value;

    explicit shm_cache(ygm::comm &c, internal_container_type &accum) : 
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
        // m_comm.cout0("alignmnt of pthread_mutex: ", alignof(pthread_mutex_t));
        // m_comm.cout0("alignment of Entry: ", alignof(Entry));
        // m_comm.cout0("size of Entry: ", sizeof(Entry));
        // m_comm.cout0("page size:", sysconf(_SC_PAGESIZE));
        //m_comm.cout0("SHM SIZE: ", SHM_SIZE);


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

        // IMPORTANT: cannot allow any other processes touch or observe mutex or any bytes associated with with
        //            until it is fully initialized.

        // initialize the beginning of the memory with a mutex
        void *base = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, 
                        MAP_SHARED, fd, 0);
        if(base == MAP_FAILED){
            perror("mapping to *shared* BIP failed\n");
            return;
        }    
        // fd is no longer needed
        close(fd);

        //m_comm.cout("Rank ", m_comm.rank(), " has finished creating a shm and mapping it");
        std::vector<m_string> BIP_filenames(m_local_size);
        auto BIP_filenames_ptr = m_comm.make_ygm_ptr(BIP_filenames);
        m_comm.barrier();
        // merge filenames to the local id 0 rank within the same physical node
        /*
            does not require pthis (distributed ygm pointer) because local rank 0 is communicating to processors 
            within the same node, eliminating the need to serialize the lambda.
            *raw pointer cannot be serialized, only ygm pointer can.
        */
        auto collect_filenames = [](auto BIP_filenames_ptr, int local_id, int node_id, int global_rank, std::string filename){
            //printf("Rank %d, Received filename %s from local rank %d, global rank %d\n", m_comm.rank(), filename.c_str(), local_id, global_rank);
            std::vector<m_string> &BIP_filenames = *BIP_filenames_ptr;
            std::snprintf(BIP_filenames.at(local_id).data, 
                        sizeof(BIP_filenames.at(local_id).data),
                        "%s",
                        filename.c_str());
            //printf("Master rank received filename %s\n", BIP_filenames.at(local_id).data);
        };
        
        int local_rank_zero = m_comm.rank() - m_local_id;
        //m_comm.cout("Sending to rank ", local_rank_zero);
        m_comm.async(local_rank_zero, collect_filenames, BIP_filenames_ptr, m_local_id, m_node_id, m_comm.rank(), filename_s);
        m_comm.barrier();

        //m_comm.cout("---- By this line, each master rank should have received their filenames ----");
        // broadcast the filenames (within the same node)
        auto broadcastBIP = [](auto BIP_filenames_ptr, std::vector<m_string> incoming_filenames){
            *BIP_filenames_ptr = std::move(incoming_filenames);
            // for(m_string &filename : *BIP_filenames_ptr){
            //     printf("Rank received filename %s\n", filename.data);
            // }
        };

        if(m_comm.rank() == local_rank_zero){
            for(int i = local_rank_zero + 1; i < local_rank_zero + m_local_size; i++){
                //m_comm.cout("Sharing filenames to rank ", i);
                m_comm.async(i, broadcastBIP, BIP_filenames_ptr, BIP_filenames);
            }
        }
        m_comm.barrier();

        //m_comm.cout("----- By this line, all processes should have received their respective node's SHMs -------");

        for(int i = 0; i < BIP_filenames.size(); i++){
            std::string str_filename(BIP_filenames[i].data);
            filename_to_BIP(m_bip_ptrs, i, str_filename, SHM_SIZE);
        }        

         // some processes may unlink before others get the chance to shm_open
        m_comm.barrier();
        //m_comm.cout("Done initializing mutex");
        shm_unlink(filename_c);

        // initialize the shared memory file with the struct data
        /*
            Could there be a better method to avoid this pre-processing? Or is this so computationally trivial that
            it does not matter?
        */

        Entry* entries = (Entry*)(header->entries);

        for (size_t i = 0; i < NUM_ENTRIES ; i++) {
            Entry* entry = &entries[i];
            entry->s_key = key_type();
            entry->s_value = -1;
        }
        m_comm.barrier();
        munmap(base, SHM_SIZE);

        //m_comm.cout("--- done with setting up the counting set ---");
    }

    ~shm_cache() {
        m_comm.barrier();
        //m_comm.log(log_level::info, "Destroying shm_cache");
    }

    // NO LONGER USING PRE-BARRIER CALLBACK
    void cache_insert(const key_type &key, const value_type &value, ygm::ygm_ptr<int> flush_count){
        int BIP_index = ygm::container::detail::hash<key_type>{}(key) % m_local_size;
        int slot = ygm::container::detail::hash<key_type>{}(key) % NUM_ENTRIES;
        YGM_ASSERT_DEBUG(BIP_index < m_local_size);
        YGM_ASSERT_DEBUG(slot < NUM_ENTRIES);

        shm_layout* header = (shm_layout*)m_bip_ptrs[BIP_index].get();
        Entry* cached_entry = &header->entries[slot];

        std::atomic<value_type> cas_value{value_type()};

        value_type expected = -1;
        while(!cas_value.atomic_compare_exchange_weak())
       
        if(cas_value.atomic_compare_exchange_weak()){ 
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
                value_cache_flush(cached_entry);
                (*flush_count)++;
                cached_entry->s_key = key;
                cached_entry->s_value = value;
            }
        }
        // if the cached value is greater than the value of int32, then flush that slot
        if(cached_entry->s_value >= std::numeric_limits<int32_t>::max() / 2){
            value_cache_flush(cached_entry);
            (*flush_count)++;
        }
    }

    /**
     * @brief Flushes a slot in the cache to the map
     * 
     * @param entry: which specific entry it should flush
     */
    void value_cache_flush(Entry* entry){

        auto key          = entry->s_key;
        auto cached_value = entry->s_value;
        YGM_ASSERT_DEBUG(cached_value > 0);

        // m_comm.cout("Pushing cached value ", cached_value, " to key ", key.x, ", ", key.y);
        m_map.async_visit(
            key,
            [](const key_type &key, value_type &partial_product, value_type to_add){
                partial_product += to_add;
            },
            cached_value
        );
        entry->s_key = key_type();
        entry->s_value = -1;
    }

    /**
     * @brief Each processor flushes its own shared memory region
     */
    void value_cache_flush_all(){
            
        shm_layout* header = (shm_layout*)m_bip_ptrs[m_local_id].get();
        Entry* entry = (Entry*)(header->entries);

        for(int i = 0; i < NUM_ENTRIES; i++){
            if((entry + i)->s_value > 0){
                value_cache_flush(entry + i);
            }
        }
    }


private:

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

        close(fd);

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
    internal_container_type                            &m_map;
    typename ygm::ygm_ptr<shm_cache>            pthis;
};
