#pragma once
#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
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

struct sum_counter{
    uint64_t sum = 0;
    uint64_t push = 0;

    template <class Archive>
    void serialize(Archive& ar) {
        ar(sum, push);
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
    explicit Sorted_COO(ygm::comm& c, ygm::container::array<Edge>& src,
                        size_t top_k,
                        std::vector<std::pair<uint64_t, size_t>> top_rows, 
                        std::vector<std::pair<uint64_t, size_t>> top_cols): m_comm(c), sorted_matrix(src), pthis(this), top_k(top_k)
                        
    {
        pthis.check(m_comm);
        row_owners.resize(m_comm.size());

        for(int i = 0; i < top_k; i++){
            this->top_rows.insert(top_rows[i].first);
            this->top_cols.insert(top_cols[i].first);
        }
        cache.reserve(top_k * top_k);
        double sort_start = MPI_Wtime();
        sorted_matrix.sort();
        double sort_end = MPI_Wtime();
        m_comm.cout0("ygm array sort time: ", sort_end - sort_start);
        
        double map_start = MPI_Wtime();

        m_comm.barrier(); 
        double map_end = MPI_Wtime();
        m_comm.cout0("row-owner map initialization time: ", map_end - map_start);

        double merge_start = MPI_Wtime();
        auto populate_row_owners = [](std::pair<uint64_t, uint64_t> min_max, int rank, auto self){
            self->row_owners[rank] = min_max;
        };

        uint64_t first = (*sorted_matrix.local_cbegin()).value.row;
        uint64_t last = -1;
        auto curr = sorted_matrix.local_cbegin();
        for(;curr != sorted_matrix.local_cend(); curr.operator++()){
            last = curr.operator*().value.row;
        }

        m_comm.async(0, populate_row_owners, 
                    std::make_pair(first, last), 
                    m_comm.rank(), pthis);
        m_comm.barrier();
        double merge_end = MPI_Wtime();
        m_comm.cout0("merge row-owner data time: ", merge_end - merge_start);

        double bc_start = MPI_Wtime();
        auto broadcast_owners = [](std::vector<std::pair<uint64_t, uint64_t>> owners, auto self){
            self->row_owners = owners;
        };
        if(m_comm.rank0()){
            m_comm.async_bcast(broadcast_owners, row_owners, pthis);
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
    ygm::container::array<Edge> &sorted_matrix;
    typename ygm::ygm_ptr<Sorted_COO> pthis;
    size_t top_k;
    // experiment 2: unordered flat map with pair<i, j> and partial product
    boost::unordered_flat_map<std::pair<uint64_t, uint64_t>, uint64_t> cache;
    // to do: create uniform generator and rmat generator
    // make a hybrid generator, e.g. 10% rmat and 90% random (uniform)
    // make the uniform generator parallel, with each rank having unique seed.
    boost::unordered_flat_set<uint64_t> top_rows;
    boost::unordered_flat_set<uint64_t> top_cols;
    // experiment 1: two more sets for i and j
    // instead of checking for double i, j
    // individually check for i and j to break out early
    std::vector<std::pair<uint64_t, uint64_t>> row_owners;

    
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
