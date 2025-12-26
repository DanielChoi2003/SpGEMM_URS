#include "sorted_coo.hpp"
using std::vector;
// module load gcc/13.3.1 openmpi
// 12/19
// 12/22 or 23
/*
    2. the count of multplications and additions should stay across nodes

    1. ygm::comm stat functions
    size of messages (bytes) should stay constant
    scope into the multiplication and addition section -> stat clear


    data movement & partitioning 
*/
/*
    Member functions defined inside the class body are implicitly inline.
*/
inline vector<int> Sorted_COO::get_owners(int source){

    vector<int> owners;
    // if the target row number is out of bound, return empty
    if(source < 0 || source > row_ptrs.size() - 2){
        return owners; 
    }
    int start = row_ptrs[source]; 
    int end = row_ptrs[source+1]; // exclusive
    while(start < end){
        owners.push_back(owner_ranks.at(start));
        start++;
    }
    return owners;
}

template<typename Fn, typename... VisitorArgs>
inline void Sorted_COO::async_visit_row(
                        int target_row, 
                        Fn user_func, 
                        VisitorArgs&... args){
        // NOTE: CAPTURING THE DISTRIBUTED CONTAINER BY REFERENCE MAY LEAD TO UNDEFINED BEHAVIOR 
        //     because the distributed container may not be in the same memory address from the remote rank (callee)'s 
        //     memory layout
    auto vlambda = 
        [user_func](const VisitorArgs... args) mutable { // lambda are const by default; args are read-only
            std::invoke(user_func, args...);
        };
    
    vector<int> owners = get_owners(target_row);
    for(int owner_rank : owners){
        //printf("Row %d is owned by rank %d\n", target_row, owner_rank);
        assert(owner_rank >= 0 && owner_rank < world.size());
        world.async(owner_rank, vlambda, args...);
    }
    
    //DO NOT CALL BARRIER HERE. PROCESSOR NEEDS TO BE ABLE TO RUN MULTIPLE TIMES.
}


// input_value, input_row, input_column, pmap

template <class Matrix, class Accumulator>
inline void Sorted_COO::spGemm(Matrix &unsorted_matrix, Accumulator &partial_accum){
    int mult_count = 0;
    int add_count = 0;
    world.cout("partitioned size: ", unsorted_matrix.local_size());
    world.stats_reset();

    shm_counting_set cache(world, partial_accum);
    auto multiplier = [&cache, &mult_count, &add_count](auto self, int input_value, int input_row, int input_column){
         // find the first Edge with matching row to input_column with std::lower_bound
        int low = 0;
        int high = self->local_size;
        int upper_bound = high;

        while (low < high) {
            int mid = low + (high - low) / 2;
            const Edge &mid_edge = self->lc_sorted_matrix[mid];

            if (mid_edge.row < input_column) { // the edge with matching row has to be to the right of mid
                low = mid + 1;
            }
            else { // the first edge with matching row has to be to the left of mid
                high = mid;
            }
        }

        // keep multiplying with the next Edge until the row number no longer matches
        for(int i = low; i < upper_bound; i++){
            const Edge &match_edge = self->lc_sorted_matrix.at(i);  
            if(match_edge.row != input_column){
                break;
            }

            // NOTE: could potentially overflow with large values
            int product = input_value * match_edge.value; // valueB * valueA;

            if(product == 0){
                continue;
            }
            mult_count++;
            // auto adder = [](const std::pair<int, int> &coord, int &partial_product, int value_add){
            //     partial_product += value_add;
            // };
            // pmap->async_visit({input_row, match_edge.col}, adder, product); // Boost's hasher complains if I use a struct

            cache.cache_insert({input_row, match_edge.col}, product, add_count);
        }   
    }; 
    
    ygm::ygm_ptr<Accumulator> pmap(&partial_accum);
    unsorted_matrix.local_for_all([&](int index, Edge &ed){
        int input_column = ed.col;
        int input_row = ed.row;
        int input_value = ed.value;
        async_visit_row(input_column, multiplier, pthis, input_value, input_row, input_column);
    });
    world.barrier();
    cache.value_cache_flush_all();
    world.barrier();
    world.stats_print();
    world.cout("number of multiplication: ", mult_count, ", number of addition: ", add_count);

}

inline void Sorted_COO::print_metadata(){
    for(int i = 0; i < metadata.size(); i++){
        world.cout("rank ", i, ": local min " , metadata.at(i).first, ", local max ", metadata.at(i).second);
    }
}

inline void Sorted_COO::print_row_owners(){
    for(int i = 0; i < row_ptrs.size() - 1; i++){
        int start = row_ptrs[i]; 
        int end = row_ptrs[i+1];
        if(start == end){
            continue;
        }
        printf("row %d is owned by rank ", i);

        while(start < end){
            int owner_rank = owner_ranks.at(start);
            if(start + 1 == end){
                printf("%d\n", owner_rank);
            }
            else{
                printf("%d, ", owner_rank);
            }
            start++;
        }
    }
}




