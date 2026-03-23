#pragma once

#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/array.hpp>
#include <ygm/container/counting_set.hpp>
#include <ygm/detail/layout.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/detail/base_misc.hpp>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <sys/mman.h>   // For shm_open, mmap
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <atomic>
#include <unistd.h>
#include <stdio.h> 
#include <unordered_set> 


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

template <typename Value>
class shm_hub{
    static_assert(std::is_trivially_copyable_v<Value>);

private:

    /*
        implicitly creates a constructor which is not trivially copyable.
        mmap gives a raw memory address and creating Entry object in the shared memory
        means that the default constructor is not called.
    */
    


    // custom string object to avoid heap pointers
    struct m_string{
        char data[256];

        template <class Archive>
        void serialize(Archive& ar) {
            ar(data);
        }
    };
    
public:

    /*
        hub_edges are merged at rank 0.
        rank 0 broadcasts hub_edges to node master ranks
        each node master creates a shared memory file and broadcasts their own file names
        to their respective non-master nodes
    */
    explicit shm_hub(ygm::comm &c, const uint32_t& topk, const uint32_t& max_num_edges,
                     std::unique_ptr<ygm::container::array<Value>>& nonhub_edges,
                     std::unique_ptr<ygm::container::counting_set<uint64_t>>& B_row_degree,
                     std::unique_ptr<ygm::container::bag<Value>>& bagbp) : 
                            m_comm(c), 
                            m_local_size(m_comm.layout().local_size()),
                            m_local_id(m_comm.layout().local_id()),
                            m_node_id(m_comm.layout().node_id()),
                            m_topk(topk){

        // FINDING HUBS
        double hub_construction_time = MPI_Wtime();
        std::vector<std::pair<uint64_t, size_t>> topk_hubs;
        topk_hubs = B_row_degree->gather_topk(m_topk,  [](const std::pair<int, size_t>& lhs, const std::pair<int, size_t>& rhs){
            if(lhs.second == rhs.second){
                return lhs.first < rhs.first;
            }
            return lhs.second > rhs.second; 
        });

        size_t cumulative_edges = 0;
        auto topk_hub_curr = topk_hubs.begin();
        while(cumulative_edges < max_num_edges && topk_hub_curr != topk_hubs.end()){
            cumulative_edges += topk_hub_curr->second;
            m_hubs.insert(topk_hub_curr->first); 
            topk_hub_curr++;
        }

        int B_row_num = B_row_degree->size();
        m_comm.cout0("There are ", B_row_num , " nodes in matrix B");
        m_comm.cout0("There are ", cumulative_edges, " hub edges out of ", bagbp->size(), " edges in matrix B");
        m_comm.barrier();
        m_num_edges = cumulative_edges;

        ygm::container::bag<Value> bag_nonhub_edges(m_comm);
        ygm::container::bag<Value> bag_hub_edges(m_comm);
        bagbp->for_all([this, &bag_nonhub_edges, &bag_hub_edges](const Value& ed){
            if(m_hubs.find(ed.row) != m_hubs.end()){ // HUB EDGE
                bag_hub_edges.async_insert(ed);
            }
            else{
                bag_nonhub_edges.async_insert(ed);
            }
        });

        nonhub_edges = std::make_unique<ygm::container::array<Value>>(m_comm, bag_nonhub_edges);
        nonhub_edges->sort();
       
        std::vector<Value> hub_edges;
        ygm::ygm_ptr<std::vector<Value>> hub_edge_ptr(&hub_edges);
        m_comm.barrier(); // to guarantee that all processors created ygm_ptr before async uses it
        hub_edge_ptr.check(m_comm);
        bag_hub_edges.gather(hub_edges, 0);
        if(m_comm.rank0()){
            for(int i = m_local_size; i < m_comm.size() ; i += m_local_size){
                m_comm.async(i, [](ygm::ygm_ptr<std::vector<Value>> hub_edge_ptr, std::vector<Value> edges){
                    *hub_edge_ptr = edges;
                }, hub_edge_ptr, hub_edges);
            }
        }
        m_comm.barrier();
        bagbp.reset();

        uint64_t SHM_SIZE = sizeof(Value) * m_num_edges;
        
        std::string filename_s = "/IP_" + std::to_string(m_node_id);
        const char *filename_c = filename_s.c_str();

        if(m_local_id == 0){

            int fd = shm_open(filename_c, O_CREAT | O_RDWR, 0666);
            if(fd == -1){
                perror("shm_open() failed\n");
                return;
            }

            if(ftruncate(fd, SHM_SIZE) == -1){
                perror("ftruncate() failed\n");
                return;
            }

            void *base = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, 
                            MAP_SHARED, fd, 0);
            if(base == MAP_FAILED){
                perror("mapping to *shared* IP failed\n");
                return;
            }    
            // fd is no longer needed
            close(fd);            
            
            // COPY THE HUB EDGES INTO THE SHARED MEM. FILE
            sort(hub_edges.begin(), hub_edges.end());
            std::memcpy(base, hub_edges.data(), SHM_SIZE);
            m_ip_ptr = std::unique_ptr<void, MMapDestructor>(base, MMapDestructor(SHM_SIZE));
            //m_comm.cout("--- Master Rank ", m_comm.rank(), " is done creating shm file ---");

        }
        m_comm.barrier(); // wait for master ranks to finish creating shm file

        if(m_local_id != 0){ // other non-master ranks map to the newly created shm file
            open_read_only(filename_s, SHM_SIZE);
            // m_comm.cout("--- Slave Rank ", m_comm.rank(), " is done mapping shm file ---");
        }
        m_comm.barrier();

        shm_unlink(filename_c);
        m_comm.cout0("Hub construction time: ", MPI_Wtime() - hub_construction_time);

    }

    const Value* get_IP_ptr(uint32_t index){
        return (Value*)(m_ip_ptr.get()) + index;
    }

    const Value* front(){
        return (Value*)(m_ip_ptr.get());
    }

    const Value* back(){
        return (Value*)(m_ip_ptr.get()) + (m_num_edges - 1);
    }

    std::unordered_set<uint64_t> get_hub_set(){
        return m_hubs;
    }

    uint32_t size(){
        return m_num_edges;
    }

    ~shm_hub() {
        m_comm.barrier();
        //m_comm.log(log_level::info, "Destroying shm_hub");
    }

private:

    struct MMapDestructor{
        size_t size;

        MMapDestructor(size_t s = 0) : size(s){} // constructor

        // when unique pointer goes out of scope, it calls the 
        // deleter function
        void operator()(void* ptr) const{
            if(ptr != MAP_FAILED){
                // unmaps the shared memory from the process's virtual address space.
                munmap(ptr, size);
            }
        }
    };

    void open_read_only(std::string filename, int size){  
        // open a file descriptor to the shared file 
        int fd = shm_open(filename.c_str(), O_RDWR, 0666);
        if (fd == -1) {
            perror("Opening received shm failed");
            return;
        }

        void *mmap_ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, 
                        MAP_SHARED, fd, 0);
        if(mmap_ptr == MAP_FAILED){
            perror("mapping to *shared* IP failed\n");
            return;
        }

        close(fd);

        m_ip_ptr = std::unique_ptr<void, MMapDestructor>(mmap_ptr, MMapDestructor(size));
    }

    // m_ is a naming convention to indicate "member"
    ygm::comm                                          &m_comm;
    // physical node size
    int                                                m_local_size = -1;
    // number of edges that will be stored in the shared memory file
    uint32_t                                           m_num_edges = -1;
    uint32_t                                           m_topk = -1;
    std::unordered_set<uint64_t>                       m_hubs;

    // Interprocess pointer to the node-local shared memeory file
    std::unique_ptr<void, MMapDestructor>              m_ip_ptr;
    int                                                m_local_id = -1;
    int                                                m_node_id = -1;
    typename ygm::ygm_ptr<shm_hub>                     pthis;
};
