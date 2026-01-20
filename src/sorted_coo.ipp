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
    auto comp_second = [](const std::pair<int, int>& lhs, int val) {
        return lhs.second < val;
    };  
   
    auto it = std::lower_bound(row_owners.begin(), row_owners.end(), source, comp_second);

    // if it is equal to the end iterator, then theres no owner
    if(it != row_owners.end()){
        int owner_rank = it - row_owners.begin();
        
        while(owner_rank < row_owners.size()){
            if(row_owners[owner_rank].first <= source){
                owners.push_back(owner_rank);
                owner_rank++;
            }
            else{
                break;
            }
        }
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
        assert(owner_rank >= 0 && owner_rank < m_comm.size());
        m_comm.async(owner_rank, vlambda, args...);
    }
    
    //DO NOT CALL BARRIER HERE. PROCESSOR NEEDS TO BE ABLE TO RUN MULTIPLE TIMES.
}


// input_value, input_row, input_column, pmap

template <class Matrix, class Accumulator>
inline void Sorted_COO::spGemm(Matrix &unsorted_matrix, Accumulator &partial_accum){
    int mult_count = 0;
    auto mult_count_ptr = m_comm.make_ygm_ptr(mult_count);
    int add_count = 0;
    auto add_count_ptr = m_comm.make_ygm_ptr(add_count);
    m_comm.stats_reset();

    m_comm.barrier();

    //proc_cache cache(m_comm, partial_accum, global_sorted_matrix.size(), global_row_number, local_nnz);
    int cache;
    auto cache_ptr = m_comm.make_ygm_ptr(cache);
    auto multiplier = [](auto pmap, auto self, 
                        int input_value, int input_row, int input_column,
                        auto cache_ptr, auto mult_count_ptr, auto add_count_ptr){
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
            (*mult_count_ptr)++;
            auto adder = [](const auto &key, auto &partial_product, auto to_add,
                            auto add_count_ptr){
                partial_product += to_add;
                (*add_count_ptr)++;
            };
            // uncomment this to test without cache
            pmap->async_visit({input_row, match_edge.col}, adder, product, add_count_ptr); // Boost's hasher complains if I use a struct

            //(*cache_ptr).cache_insert({input_row, match_edge.col}, product);
        }   
    }; 
    
    ygm::ygm_ptr<Accumulator> pmap(&partial_accum);
    unsorted_matrix.local_for_all([&](int index, Edge &ed){
        int input_column = ed.col;
        int input_row = ed.row;
        int input_value = ed.value;
        async_visit_row(input_column, multiplier, 
                        pmap, pthis, input_value, input_row, input_column,
                        cache_ptr, mult_count_ptr, add_count_ptr);
    });
    m_comm.barrier();
    //cache.cache_flush_all();
    m_comm.stats_print();
    //cache.cache_print();
    //m_comm.cout("number of multiplication: ", mult_count, ", number of addition: ", add_count);

}

inline void Sorted_COO::print_row_owners(){
}




