#pragma once

#include "shm_hub/shm_hub.h"
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

private:
    std::unique_ptr<ygm::container::array<Edge>> nonhub_edges;         // globally sorted nonhub edges
    shm_hub<Edge> SHM_HUB;

public:

    /*
        @brief Initializes the ygm::container::array member with a ygm::container::bag provided by the user.

        @param ygm::comm&: communicator object
        @param ygm::container::array<Edge>& src: array that will be sorted in the constructor.
    */
    explicit Sorted_COO(ygm::comm& c, const uint32_t& topk, const uint32_t& max_hub_edges,
                        std::unique_ptr<ygm::container::counting_set<uint64_t>>& matrix_B_row_degree,
                        std::unique_ptr<ygm::container::bag<Edge>>& matrix_B_bag
                        ): 
                        m_comm(c), pthis(this),
                        SHM_HUB(c, topk, max_hub_edges, nonhub_edges, matrix_B_row_degree, matrix_B_bag)         
    {
        pthis.check(m_comm);
        nonhub_row_owners.resize(m_comm.size());
        
        double map_start = MPI_Wtime();
        
        //  NONHUB EDGE ROW OWNERS
        uint64_t first = (*nonhub_edges->local_cbegin()).value.row;
        uint64_t last = -1;
        auto it = nonhub_edges->local_cbegin();
        for(;it != nonhub_edges->local_cend(); it.operator++()){
            last = it.operator*().value.row;
        }

        // POPULATING THE ROW PTRS FOR NONHUB EDGES
        // plus one to get the range of rows, additional plus one for row ptr's last index
        row_ptrs.resize(last - first + 2);
        offset = first;
        auto curr = nonhub_edges->local_cbegin();
        uint64_t row_index = 0;
        uint64_t ptr_index = 0; // index of the row ptrs 
        row_ptrs[ptr_index] = row_index;
        for(;curr != nonhub_edges->local_cend(); ++curr){
            while((offset + ptr_index) != (*curr).value.row){
                ptr_index++;
                row_ptrs[ptr_index] = row_index;
            }
            row_index++;
        }
        row_ptrs.back() = row_index; // last index + 1

        uint64_t hub_first = SHM_HUB.front()->row;
        uint64_t hub_last = SHM_HUB.back()->row;
        // POPULATING THE ROW PTRS FOR HUB EDGES
        hub_row_ptrs.resize(hub_last - hub_first + 2);
        hub_offset = hub_first;
        row_index = 0;
        ptr_index = 0; // index of the row ptrs 
        hub_row_ptrs[ptr_index] = row_index;
        for(uint32_t i = 0; i < SHM_HUB.size(); i++){
            while((hub_offset + ptr_index) != SHM_HUB.get_IP_ptr(i)->row){
                ptr_index++;
                hub_row_ptrs[ptr_index] = row_index;
            }
            row_index++;
        }
        hub_row_ptrs.back() = row_index; // last index + 1
    
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
    void spGemm(Matrix &matrix_A, Accumulator &partial_accum, uint64_t& total_mult);


private:
    ygm::comm &m_comm;                                                 // store the communicator. Hence the &
    typename ygm::ygm_ptr<Sorted_COO> pthis;
  
    double owner_search_time = 0;
    /*
     * nonhub edges' rows are diverse, which can be expensive to store in a map
     * ISSUE: nonhub_row_owners metadata will most likely direct it to the wrong owner if the row is a hub row
     * SOLUTION: determine ahead whether a row is a hub row or not before searching for owners
    */
    std::vector<std::pair<uint64_t, uint64_t>> nonhub_row_owners; // for nonhub edges
    std::vector<uint64_t> row_ptrs;
    uint64_t offset;
    /* 
     * hub edges' rows are not diverse as nonhub edges; they are concentrated in few rows
     * hence, it may be more appropriate to use a map for simplicity and speed
    */
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
