#pragma once
#include "proc_cache/proc_cache.hpp"
#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/array.hpp>
#include <ygm/container/set.hpp>
#include <ygm/container/counting_set.hpp>
#include <ygm/io/csv_parser.hpp>
#include <ygm/container/bag.hpp>
#include <cereal/types/unordered_set.hpp> // to support serializing unordered set
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct map_key{
    int x;
    int y;

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
    int row;
    int col;
    int value;
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
    explicit Sorted_COO(ygm::comm& c, ygm::container::array<Edge>& src): world(c), global_sorted_matrix(src), pthis(this) {
        pthis.check(world);

        double sort_start = MPI_Wtime();
        src.sort();
        double sort_end = MPI_Wtime();
        world.cout0("ygm array sort time: ", sort_end - sort_start);
        
        local_size = src.local_size();
        // {key, value}: {row number, set of owners}

        double map_start = MPI_Wtime();
        ygm::container::map<int, std::unordered_set<int>> global_row_owners(world);

        src.for_all([this, &global_row_owners](int index, Edge &ed){
            global_row_owners.async_visit(ed.row, [](int key, std::unordered_set<int> &owners, int rank){
                owners.insert(rank);
            }, world.rank());
            lc_sorted_matrix.push_back(ed);
        });
        world.barrier(); 
        double map_end = MPI_Wtime();
        world.cout0("row-owner map initialization time: ", map_end - map_start);


        double merge_start = MPI_Wtime();
        auto merge_row_owners = [](int row, std::unordered_set<int> owners, auto self){
            self->row_owners[row] = owners;
        };
        global_row_owners.for_all([this, merge_row_owners](int const &key, std::unordered_set<int> const &owners){
            world.async(0, merge_row_owners, key, owners, pthis);
        });
        world.barrier();
        double merge_end = MPI_Wtime();
        world.cout0("merge row-owner data time: ", merge_end - merge_start);

        double bc_start = MPI_Wtime();
        auto broadcast_owners = [](std::unordered_map<int, std::unordered_set<int>> owners, auto self){
            self->row_owners = owners;
        };
        if(world.rank0()){
            world.async_bcast(broadcast_owners, row_owners, pthis);
        }
        world.barrier();
        double bc_end = MPI_Wtime();
        world.cout0("broadcast row-owner data time: ", bc_end - bc_start);

    }

    void print_row_owners();

    /*
        @brief 
            gets the owners of the row number that matches to the given argument "source".
    
        @param source: the number of the row number 
    */
    std::vector<int> get_owners(int source);

   
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
    void async_visit_row(int target_row, Fn user_func, VisitorArgs&... args);


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
    std::unordered_map<int, std::unordered_set<int>> row_owners;
    ygm::container::array<Edge>& global_sorted_matrix;
    std::vector<Edge> lc_sorted_matrix;
    int local_size = -1;
    ygm::comm &world;                            // store the communicator. Hence the &
    typename ygm::ygm_ptr<Sorted_COO> pthis;
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
