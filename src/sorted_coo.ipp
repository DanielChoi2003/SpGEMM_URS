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

    #define CACHE

    auto multiplier = [](auto pmap, auto self, 
                        int input_value, int input_row, int input_column,
                        auto mult_count_ptr, auto add_count_ptr){
         // find the first Edge with matching row to input_column with std::lower_bound
        int low = self->sorted_matrix.partitioner.local_start();
        int high = low + self->sorted_matrix.partitioner.local_size();
        int upper_bound = high;

        while (low < high) {
            int mid = low + (high - low) / 2;

            Edge mid_edge = {};
            // local visit expects a global index. internally, converts it into a local index: 0 to local size
            self->sorted_matrix.local_visit(mid, [&mid_edge](int index, Edge &edge){
                mid_edge = edge;
            });

            if (mid_edge.row < input_column) { // the edge with matching row has to be to the right of mid
                low = mid + 1;
            }
            else { // the first edge with matching row has to be to the left of mid
                high = mid;
            }
        }

        // keep multiplying with the next Edge until the row number no longer matches
        for(int i = low; i < upper_bound; i++){
            Edge match_edge = {};  
            self->sorted_matrix.local_visit(i, [&match_edge](int index, Edge &edge){
                match_edge = edge;
            });
            if(match_edge.row != input_column){
                break;
            }

            // NOTE: could potentially overflow with large values
            int product = input_value * match_edge.value; // valueB * valueA;

            if(product == 0){
                continue;
            }
            (*mult_count_ptr)++;
            auto adder = [](const auto &key, auto &sum_counter_pair, auto to_add,
                            auto add_count_ptr){
                sum_counter_pair.sum += to_add;
                sum_counter_pair.push++;
                (*add_count_ptr)++;
            };

            #ifdef CACHE
                if(self->top_rows.count(input_row) && self->top_cols.count(match_edge.col)){
                    auto [it, inserted] = self->cache.try_emplace({input_row, match_edge.col}, product);
                    if (!inserted) {
                        it->second += product;
                    }
                }
                else{
                    pmap->async_visit({input_row, match_edge.col}, adder, product, add_count_ptr); // Boost's hasher complains if I use a struct
                }
            #endif

            #ifndef CACHE
                pmap->async_visit({input_row, match_edge.col}, adder, product, add_count_ptr); // Boost's hasher complains if I use a struct
            #endif

        }   
    }; 
    
    ygm::ygm_ptr<Accumulator> pmap(&partial_accum);
    unsorted_matrix.local_for_all([&](int index, Edge &ed){
        int input_column = ed.col;
        int input_row = ed.row;
        int input_value = ed.value;
        async_visit_row(input_column, multiplier, 
                        pmap, pthis, input_value, input_row, input_column,
                        mult_count_ptr, add_count_ptr);
    });
    m_comm.barrier();
    #ifdef CACHE
    auto adder = [](const auto &key, auto &sum_counter_pair, auto to_add){
        sum_counter_pair.sum += to_add;
        sum_counter_pair.push++;
    };
    for (auto& [key, value] : cache) {
        pmap->async_visit({key.first, key.second}, adder, value);
    }
    m_comm.barrier();
    #endif
    m_comm.stats_print();
    //m_comm.cout("number of multiplication: ", mult_count, ", number of addition: ", add_count);

}





