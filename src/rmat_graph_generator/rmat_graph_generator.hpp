#pragma once

#include <cmath>
#include <fstream>
#include "../../include/ygm-gctc/rmat_edge_generator.hpp"
#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/counting_set.hpp>

// struct Edge{
//     uint64_t row;
//     uint64_t col;
//     bool operator<(const Edge& B) const{ // does not modify the content
//         if (row != B.row) return row < B.row; // first, sort by row
//         return col < B.col; // if rows are equal, sort by column
      
//     }

//     template <class Archive>
//     void serialize( Archive & ar )
//     {
//         ar(row, col);
//     }
// };

template<typename edge_data_type>
class rmat_graph_generator{
public:

    rmat_graph_generator(ygm::comm &c, ygm::container::bag<edge_data_type> &rmat_edges,
                        ygm::ygm_ptr<ygm::container::counting_set<uint64_t>> top_index = nullptr) 
                        : m_comm(c), rmat_edges(rmat_edges), top_index(top_index){
                        }

    void generate_rmat_edges(uint64_t scale, uint64_t total_edges, 
                            double a, double b, double c, double d, 
                            double alpha, bool want_top_row = false,
                            bool want_top_col = false, bool transpose = false){

        uint64_t edges_per_rank = total_edges / m_comm.size();
        uint64_t start = m_comm.rank() * edges_per_rank;
        uint64_t end   = (m_comm.rank() == m_comm.size() - 1) ? total_edges : start + edges_per_rank;

        std::mt19937 rng(m_comm.rank());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        std::uniform_int_distribution<uint64_t> unif(0, (1ULL << scale) - 1);

        for (uint64_t e = start; e < end; ++e) {
            uint64_t u, v;

            if (coin(rng) < alpha) {
                // RMAT edge
                u = 0; v = 0;
                for (int i = 0; i < scale; ++i) {
                    u <<= 1; v <<= 1;
                    double r = dist(rng);
                    if (r < a) {}
                    else if (r < a+b) v |= 1;
                    else if (r < a+b+c) u |= 1;
                    else { u |= 1; v |= 1; }
                }
            } else {
                // Uniform edge
                u = unif(rng);
                v = unif(rng);
            }

            // Optional but important
            u = gctc::detail::hash_nbits(u, scale);
            v = gctc::detail::hash_nbits(v, scale);

            if(transpose){
                rmat_edges.async_insert({v, u});
                if(want_top_row){
                    top_index->async_insert(v);
                }
                else if(want_top_col){
                    top_index->async_insert(u);
                }
            }
            else{
                rmat_edges.async_insert({u, v});
                if(want_top_row){
                    top_index->async_insert(u);
                }
                else if(want_top_col){
                    top_index->async_insert(v);
                }
            }
        
        }
    }     
    
private:
    ygm::comm& m_comm;
    ygm::container::bag<edge_data_type> &rmat_edges;
    ygm::ygm_ptr<ygm::container::counting_set<uint64_t>> top_index;

};