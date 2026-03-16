#pragma once
#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/array.hpp>
#include <ygm/container/set.hpp>
#include <ygm/container/counting_set.hpp>
#include <cereal/types/unordered_set.hpp> // to support serializing unordered set
#include <boost/unordered/unordered_flat_map.hpp>
#include <ygm/container/detail/block_partitioner.hpp> // for local_start() and local_end()
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <vector>
#include <unordered_set>
#include <unordered_map>

struct map_key{
    uint64_t x;
    uint64_t y;

    bool operator==(const map_key& other) const {
        return x == other.x && y == other.y;
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(x, y);
    }
};

/*
    std::pair is not trivially copyable -> need to use struct ->
    requires custom hashing for the struct as std::pair is no longer
    used
*/
std::size_t hash_value(map_key const& key) {
  std::size_t seed = 0;
  boost::hash_combine(seed, key.x);
  boost::hash_combine(seed, key.y);
  return seed;
}

struct Edge{
    uint64_t row;
    uint64_t col;
    uint64_t value;
    bool operator<(const Edge& B) const{ // does not modify the content
        if (row != B.row) return row < B.row; // first, sort by row
        if (col != B.col) return col < B.col; // if rows are equal, sort by column
        return value < B.value; // lastly sort by value
    }

    template <class Archive>
    void serialize( Archive & ar )
    {
        ar(row, col, value);
    }
};


class Sorted_COO{

public:

    /*
        @brief Initializes the ygm::container::array member with a ygm::container::bag provided by the user.

        @param ygm::comm&: communicator object
        @param ygm::container::array<Edge>& src: array that will be sorted in the constructor.
    */
    explicit Sorted_COO(ygm::comm& c, ygm::container::array<Edge>& nonhub_edges,
                        ygm::container::array<Edge>& hub_edges, std::unordered_set<uint64_t> &B_hubs): 
                        m_comm(c), nonhub_edges(nonhub_edges), 
                        hub_edges(hub_edges), pthis(this), B_hub_rows(B_hubs)            
    {
        pthis.check(m_comm);
        nonhub_row_owners.resize(m_comm.size());

        double sort_start = MPI_Wtime();
        nonhub_edges.sort();
        if(hub_edges.size() > 1){
            hub_edges.sort();
        }
        m_comm.barrier(); 
        m_comm.cout0("ygm array sort time: ", MPI_Wtime() - sort_start);
        
        double map_start = MPI_Wtime();

        hub_edges.for_all([this](const int &index, const Edge &ed){
            // INSERT THE OWNER RANK IF IT DOES NOT EXIST
            this->hub_row_owners[ed.row].insert(this->m_comm.rank());
        });

        if(!hub_row_owners.empty()){
            // flatten before sending
            // unordered_map is not serializable
            std::vector<std::pair<uint64_t, std::vector<uint32_t>>> flat;
            for(auto& [key, inner_set] : hub_row_owners){
                flat.push_back({key, std::vector<uint32_t>(inner_set.begin(), inner_set.end())});
            }
            
            if(m_comm.rank() != 0){
                auto sendHubOwner = [](auto self,
                                    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> hub_other_owners){
                    for(auto& [key, vec] : hub_other_owners){
                        self->hub_row_owners[key].insert(vec.begin(), vec.end());
                    }           
                };
                m_comm.async(0, sendHubOwner, pthis, flat);
            }
            m_comm.barrier();

            if(m_comm.rank0()){
                flat.clear(); // flatten it again for rank 0
                for(auto& [key, inner_set] : hub_row_owners){
                    flat.push_back({key, std::vector<uint32_t>(inner_set.begin(), inner_set.end())});
                }
                auto broadcastHubOwner = [](auto self,
                                            std::vector<std::pair<uint64_t, std::vector<uint32_t>>> hub_all_owners){
                    for(auto& [key, vec] : hub_all_owners){
                        self->hub_row_owners[key].insert(vec.begin(), vec.end());
                    }  
                };
                m_comm.async_bcast(broadcastHubOwner, pthis, flat);
            }
        }
        m_comm.barrier();

        //  NONHUB EDGE ROW OWNERS
        uint64_t first = (*nonhub_edges.local_cbegin()).value.row;
        uint64_t last = -1;
        auto it = nonhub_edges.local_cbegin();
        for(;it != nonhub_edges.local_cend(); it.operator++()){
            last = it.operator*().value.row;
        }

        //m_comm.cout("Before accessing hub edge iterator");
        // HUB EDGE ROW OWNERS
        uint64_t hub_first = (*hub_edges.local_cbegin()).value.row;
        uint64_t hub_last = -1;
        it = hub_edges.local_cbegin();
        for(;it != hub_edges.local_cend(); it.operator++()){
            hub_last = it.operator*().value.row;
        }
        //m_comm.cout("After accessing hub edge iterator");

        // POPULATING THE ROW PTRS FOR NONHUB EDGES
        // plus one to get the range of rows, additional plus one for row ptr's last index
        row_ptrs.resize(last - first + 2);
        offset = first;
        auto curr = nonhub_edges.local_cbegin();
        uint64_t row_index = 0;
        uint64_t ptr_index = 0; // index of the row ptrs 
        row_ptrs[ptr_index] = row_index;
        for(;curr != nonhub_edges.local_cend(); ++curr){
            while((offset + ptr_index) != (*curr).value.row){
                ptr_index++;
                row_ptrs[ptr_index] = row_index;
            }
            row_index++;
        }
        row_ptrs.back() = row_index; // last index + 1

        // POPULATING THE ROW PTRS FOR HUB EDGES
        hub_row_ptrs.resize(hub_last - hub_first + 2);
        hub_offset = hub_first;
        curr = hub_edges.local_cbegin();
        row_index = 0;
        ptr_index = 0; // index of the row ptrs 
        hub_row_ptrs[ptr_index] = row_index;
        for(;curr != hub_edges.local_cend(); ++curr){
            while((hub_offset + ptr_index) != (*curr).value.row){
                ptr_index++;
                hub_row_ptrs[ptr_index] = row_index;
            }
            row_index++;
        }
        hub_row_ptrs.back() = row_index; // last index + 1
        // for(int i = 0; i < hub_row_ptrs.size() - 1; i++){
        //     m_comm.cout("For row ", i + hub_offset, ", begin: ", hub_row_ptrs[i], " and end: ", hub_row_ptrs[i + 1]);
        // }

        m_comm.barrier(); 
        double map_end = MPI_Wtime();
        m_comm.cout0("row-owner map initialization time: ", map_end - map_start);

        double merge_start = MPI_Wtime();
        auto populate_row_owners = [](std::pair<uint64_t, uint64_t> min_max, int rank, auto self){
            self->nonhub_row_owners[rank] = min_max;
        };
        m_comm.async(0, populate_row_owners, 
                    std::make_pair(first, last), 
                    m_comm.rank(), pthis);

        m_comm.barrier();
        double merge_end = MPI_Wtime();
        m_comm.cout0("merge row-owner data time: ", merge_end - merge_start);

        double bc_start = MPI_Wtime();
        auto broadcast_owners = [](std::vector<std::pair<uint64_t, uint64_t>> owners, auto self){
            self->nonhub_row_owners = owners;
        };
        if(m_comm.rank0()){
            m_comm.async_bcast(broadcast_owners, nonhub_row_owners, pthis);
        }
        m_comm.barrier();
        double bc_end = MPI_Wtime();
        m_comm.cout0("broadcast row-owner data time: ", bc_end - bc_start);

    }

    void print_row_owners();

    /*
        @brief 
            gets the owners of the row number that matches to the given argument "source".
    
        @param source: the number of the row number 
    */
    std::vector<uint64_t> get_owners(uint64_t source);

   
    /**
        @brief
            finds the set of owners (ranks) that contains elements with the matching row number.
            The caller of this function calls the owner(s) by providing the column number, row number, and
            value operands to multiply with.
            The callee will find the index of the first occurring element with a matching row number.
            The callee will multiply the found elements with the given value and store the partial products in
            [given row number, the multiplied element's column number].



        @param input_column: incoming column number. Will be multipled with a value that has a matching row number.
        @param input_row: incoming number row number. Used to determine the partial product's index.
        @param input_value: what will be multiplied with.

        @return none
    */
    template<typename Fn, typename... VisitorArgs>
    void async_visit_row(uint64_t target_row, Fn user_func, VisitorArgs&... args);


    /*
        @brief 
            Matrix A (unsorted) starts the matrix multiplication. Intermediate partial products are stored
            in the Accumulator class, which is a ygm::container::map for now.
            This function calls async_visit_row();

        @param Matrix matrix_A: unsorted matrix that starts the sparse multiplication. Traverses column-by-column.
        @param Accumulator C: distributed map that stores the partial products
    */
    template <class Matrix, class Accumulator>
    void spGemm(Matrix &matrix_A, Accumulator &partial_accum);


private:
    ygm::comm &m_comm;                            // store the communicator. Hence the &
    ygm::container::array<Edge> &nonhub_edges;         // globally sorted nonhub edges
    ygm::container::array<Edge> &hub_edges;            // globally sorted hub edges
    typename ygm::ygm_ptr<Sorted_COO> pthis;
  
    double owner_search_time = 0;
    /*
     * nonhub edges' rows are diverse, which can be expensive to store in a map
     * ISSUE: nonhub_row_owners metadata will most likely direct it to the wrong owner if the row is a hub row
    */
    std::vector<std::pair<uint64_t, uint64_t>> nonhub_row_owners; // for nonhub edges
    std::vector<uint64_t> row_ptrs;
    uint64_t offset;
    /* 
     * hub edges' rows are not diverse as nonhub edges; they are concentrated in few rows
     * hence, it may be more appropriate to use a map for simplicity and speed
    */
    std::unordered_set<uint64_t> &B_hub_rows;
    std::unordered_map<uint64_t, std::unordered_set<uint32_t>> hub_row_owners;
    std::vector<uint64_t> hub_row_ptrs;
    uint64_t hub_offset;

    // KEEP TRACK OF EACH RANK'S # OF MULTIPLICATION AND ADDITION
    uint64_t mult_count = 0, add_count = 0;
};


// including the ipp file here removes the need to add it in add_ygm_executable()
#include "sorted_coo.ipp"


/*
    1. would having another YGM container in the class lead to too much overhead? Does it create an entirely new copy
        or use the local data to create a partial copy. Cannot determine the behavior of multiple ranks calling the same
        constructor function.

    2. When using lambda function, does captured variable always refer to the callee's or caller's?
        Answer:
            Assuming that & uses the caller's memory address

    3. using "this" pointer leads to segmentation fault.
        Theory is that the memory address contained in "this" pointer may be different from the callee's "this" pointer's memory
        address, thus leading to segmentation fault.
    
    
    
    undefined reference to sorted_coo.ipp. 
        Solution: adding inline to defined functions and adding #include "sorted_coo.ipp" at the end of "sorted_coo.hpp"

*/
