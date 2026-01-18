#pragma once
#include "shm_counting_set/shm_counting_set.h"
#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/array.hpp>
#include <ygm/container/counting_set.hpp>
#include <ygm/io/csv_parser.hpp>
#include <ygm/container/bag.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <vector>

using map_index = std::pair<int, int>;

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
    explicit Sorted_COO(ygm::comm& c, ygm::container::array<Edge>& src): world(c), pthis(this) {
        double sort_start = MPI_Wtime();
        src.sort();
        double sort_end = MPI_Wtime();
        world.cout0("ygm array sort time: ", sort_end - sort_start);
        // creation of a container requires all ranks to be present
        /*
            temporary set to keep track of nonzero rows.
            To be used when checking middle row between min and max rows.
        */
        ygm::container::counting_set<int> nonzero_rows(world);
        pthis.check(world);
        /*
            index = rank number
            pair<minimum row number, maximum row number> the rank holds
            get the minimum and maximum row number that each processor holds

            to gather/merge, you can either use:
            1. a distributed data structure, then call gather on it
            2. use rank 0's local data structure, call async to rank 0, 
                insert the data into index that matches the caller rank's id.
                Then rank 0 broadcasts to all other ranks
        */
       // does comm::size() implicitly call barrier()?
       // array.size() contains a barrier()
        int num_of_processors = world.size();
        metadata.resize(num_of_processors);

        double nnz_start = MPI_Wtime();
        src.local_for_all([this, &nonzero_rows](int index, Edge ed){
            nonzero_rows.async_insert(ed.row);
            lc_sorted_matrix.push_back(ed);
        });
        double nnz_end = MPI_Wtime();
        world.cout0("nonzero row construction time: ", nnz_end - nnz_start);


        local_size = src.local_size();
        //printf("rank %d has %d nonzero elements\n", world.rank(), local_size);
        // it may have zero nnz elements
        if(!lc_sorted_matrix.empty()){
            local_min = lc_sorted_matrix.front().row;
            local_max = lc_sorted_matrix.back().row;
        }
        else{
            local_min = INT_MAX;
            local_max = INT_MIN;
        }
        

        double gather_bc_start = MPI_Wtime();
        //printf("rank %d: local min %d, local max %d\n", world.rank(), local_min, local_max);
        auto mt_inserter = [](int rank_num, std::pair<int, int> min_max, auto pCoo){
            //printf("Inserting local min %d and local max %d at index %d\n", min_max.first, min_max.second, rank_num);
            pCoo->metadata.at(rank_num) = min_max;
        };
        // if local_min is greater than local_max (meaning the rank does not own any elements), then don't add it to metadata
        if(local_min <= local_max){
            world.async(0, mt_inserter, world.rank(), std::make_pair(local_min, local_max), pthis);
        }
        world.barrier();

        //now broadcast it to all other ranks
        auto broadcastMetadata = [this](std::vector<std::pair<int, int>> incoming_metadata, auto pCOO){
            pCOO->metadata = incoming_metadata;
        };

        if(world.rank0()){
            world.async_bcast(broadcastMetadata, metadata, pthis);
        }
        world.barrier(); 
        double gather_bc_end = MPI_Wtime();
        world.cout0("Gather and broadcast metadata time: ", gather_bc_end - gather_bc_start);


        double row_ptrs_start = MPI_Wtime();
        /*
            problem: missing gaps (row number that is not owned by any rank and not between rank's min & max) will misalign
                    the row_ptrs array.
                    Also if the row number does not start with zero, it also gets misaligned by one index.
        */
        int global_min = metadata.empty() ? 0 : metadata.front().first;
        // pad the row_ptrs array, so the first row number can access the correct owners (uses index as the row number)
        for(int k = 0; k < global_min; k++){
            row_ptrs.push_back(owner_ranks.size());
        }
        int previous_row = global_min - 1;
        for(int i = 0; i < metadata.size(); i++){ // i is the owner rank
            int current_row = metadata[i].first;

            /*
                fill gaps between previous rank's max row number
                and the current rank's min row number.
                ex: min max
                0: [0, 3]
                1: [6, 9]
                missing row numbers: 4, 5
            */ 
            for(int j = previous_row + 1; j < current_row; j++){
                row_ptrs.push_back(owner_ranks.size());
            }
            while(current_row <= metadata[i].second){
                if(previous_row != current_row){
                    row_ptrs.push_back(owner_ranks.size());
                }
                if(nonzero_rows.count(current_row) != 0){
                    owner_ranks.push_back(i);
                }
                previous_row = current_row;
                current_row++;
            }
        }
        row_ptrs.push_back(owner_ranks.size());
        // don't forget a barrier here since spgemm relies on these metadata.
        world.barrier(); 
        double row_ptrs_end = MPI_Wtime();
        world.cout0("row ptrs and owner rank vector initialization time: ", row_ptrs_end - row_ptrs_start);

    }

    /*
        @brief 
            prints each rank's metadata vector. A test case function to ensure that 
            each rank contains the same global data.
    */
    void print_metadata();

    void print_row_owners();

    void print_row_ptrs(){
        printf("row_ptrs: ");
        for(int i = 0; i < row_ptrs.size(); i++){
            if(i == 0){
                printf("[ %d, ", row_ptrs.at(i));
            }
            else if (i == row_ptrs.size() - 1){
                printf("%d ]\n", row_ptrs.at(i));
            }
            else{
                printf("%d, ", row_ptrs.at(i));
            }
        }
    }

    void print_owner_ranks(){
        printf("owner_ranks: ");
        for(int i = 0; i < owner_ranks.size(); i++){
            if(i == 0){
                printf("[ %d, ", owner_ranks.at(i));
            }
            else if (i == owner_ranks.size() - 1){
                printf("%d ]\n", owner_ranks.at(i));
            }
            else{
                printf("%d, ", owner_ranks.at(i));
            }
        }
    }
    /*
        @brief 
            gets the owners of the row number that matches to the given argument "source".
    
        @param source: the number of the row number 
    */
    std::vector<int> get_owners(int source);

   
    /*
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
    /*
        contains each processor's min and max source number (row number)
    */
    std::vector<std::pair<int, int>> metadata;
    /*
        CSR data structure for O(1) lookup
    */
    std::vector<int> owner_ranks;
    std::vector<int> row_ptrs;
    std::vector<int> nonzero_rows;


    int local_size = -1;
    int local_min = -1;
    int local_max = -1;
    ygm::comm &world;                            // store the communicator. Hence the &
    std::vector<Edge> lc_sorted_matrix;  // store the local sorted matrix
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
