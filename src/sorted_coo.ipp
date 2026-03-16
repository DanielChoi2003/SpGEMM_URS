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

inline vector<uint64_t> Sorted_COO::get_owners(uint64_t source){

    double st = MPI_Wtime();
    vector<uint64_t> owners;
    // FIRST CHECK IF ITS A HUB ROW
    if(B_hub_rows.find(source) != B_hub_rows.end()){ 
        vector<uint64_t> hub_owners(hub_row_owners[source].begin(), hub_row_owners[source].end());
        return hub_owners;
    }
    auto comp_second = [](const std::pair<uint64_t, uint64_t>& lhs, uint64_t val) {
        return lhs.second < val;
    };  
   
    auto it = std::lower_bound(nonhub_row_owners.begin(), nonhub_row_owners.end(), source, comp_second);

    // if it is equal to the end iterator, then theres no owner
    if(it != nonhub_row_owners.end()){
        uint64_t owner_rank = it - nonhub_row_owners.begin();
        
        while(owner_rank < nonhub_row_owners.size()){
            if(nonhub_row_owners[owner_rank].first <= source){
                owners.push_back(owner_rank);
                owner_rank++;
            }
            else{
                break;
            }
        }
    }
    st = MPI_Wtime() - st;
    owner_search_time += st;
    return owners;
}

template<typename Fn, typename... VisitorArgs>
inline void Sorted_COO::async_visit_row(
                        uint64_t target_row, 
                        Fn user_func, 
                        VisitorArgs&... args){
        // NOTE: CAPTURING THE DISTRIBUTED CONTAINER BY REFERENCE MAY LEAD TO UNDEFINED BEHAVIOR 
        //     because the distributed container may not be in the same memory address from the remote rank (callee)'s 
        //     memory layout
    auto vlambda = 
        [user_func](const VisitorArgs... args) mutable { // lambda are const by default; args are read-only
            std::invoke(user_func, args...);
        };
    
    vector<uint64_t> owners = get_owners(target_row);
    for(uint64_t owner_rank : owners){
        //printf("Row %d is owned by rank %d\n", target_row, owner_rank);
        assert(owner_rank >= 0 && owner_rank < m_comm.size());
        m_comm.async(owner_rank, vlambda, args...);
    }
}


// input_value, input_row, input_column, pmap

template <class Matrix, class Accumulator>
inline void Sorted_COO::spGemm(Matrix &unsorted_matrix, Accumulator &partial_accum){
    m_comm.stats_reset();

    auto multiplier = [](auto pmap, auto self, 
                        uint64_t input_value, uint64_t input_row, uint64_t input_column){
        uint64_t loc;
        uint64_t global_offset;
        uint64_t start;
        uint64_t end; // EXCLUSIVE
        ygm::container::array<Edge> *edges;
        if(self->B_hub_rows.find(input_column) != self->B_hub_rows.end()){ // IF HUB ROW
            loc = input_column - self->hub_offset;
            global_offset = self->hub_edges.partitioner.local_start();
            start = global_offset + self->hub_row_ptrs[loc];
            end = global_offset + self->hub_row_ptrs[loc + 1];
            edges = &self->hub_edges;
        }
        else{
            loc = input_column - self->offset;
            global_offset = self->nonhub_edges.partitioner.local_start();
            start = global_offset + self->row_ptrs[loc];
            end = global_offset + self->row_ptrs[loc + 1];
            edges = &self->nonhub_edges; 
        }


        for(; start < end; start++){
            Edge match_edge = {};  
            edges->local_visit(start, [&match_edge](uint64_t index, Edge &edge){
                match_edge = edge;
            });
            
            // NOTE: could potentially overflow with large values
            uint64_t product = input_value * match_edge.value; // valueB * valueA;
            self->mult_count++;
            if(product == 0){
                continue;
            }
            auto adder = [](const auto &key, auto &value, auto to_add, auto self){
                value += to_add;
                self->add_count++;
            };
            // is there a way to locally store and then merge it later? to reduce the number of async messages
            pmap->async_visit({input_row, match_edge.col}, adder, product, self); // Boost's hasher complains if I use a struct
        } 
    }; 
    
    ygm::ygm_ptr<Accumulator> pmap(&partial_accum);
    // URGENT:
    // for(auto &ed : unsorted_matrix)
    //    for every X counter,
    //    m_comm.async_barrier(); interal buffer may be overflowing due to flooding
    size_t counter = 0;
    // ygm may be returning Edge by value, not by reference. hence, non-const cannot be bind to it.
    for(auto const &ptr : unsorted_matrix){
        Edge const &ed = ptr.value;
        uint64_t input_column = ed.col;
        uint64_t input_row = ed.row;
        uint64_t input_value = ed.value;
        async_visit_row(input_column, multiplier, 
                        pmap, pthis, input_value, input_row, input_column);
        counter++;
        if(counter == 100000){
            //m_comm.async_barrier();
            counter = 0;
        }
    }
    m_comm.barrier(); 

    uint64_t mult_total = ygm::sum(mult_count, m_comm);
    uint64_t mult_max  = ygm::max(mult_count, m_comm);
    uint64_t mult_avg = mult_total / m_comm.size();
    m_comm.cout0("Multiplication Count Max: ", mult_max, ", Multiplication Count Average: ", mult_avg);
    m_comm.stats_print();
}





