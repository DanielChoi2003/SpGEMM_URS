#include "rmat_graph_generator.hpp"
#include <ygm/container/array.hpp>



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

int main(int argc, char **argv){

    ygm::comm world(&argc, &argv);
    static ygm::comm &s_world = world;


    ygm::container::bag<Edge> rmat_bag(world);
    ygm::container::counting_set<uint64_t> top_rows(world);
    auto top_row_ptr = world.make_ygm_ptr(top_rows);

    rmat_graph_generator rmat_gen(world, rmat_bag, top_row_ptr);
    rmat_gen.generate_rmat_edges(25, 80000000, 0.74, 0.06, 0.1, 0.1, 0.8, true);

    world.barrier();

    ygm::container::array<Edge> rmat_edges(world, rmat_bag);

    // for(auto edge_iterator : rmat_edges){
    //     Edge &ed = edge_iterator.value;
    //     s_world.cout("row: ", ed.row, ", column: ", ed.col, ", value: ", ed.value);
    // }

    size_t k = 25;
    auto comp_count = [](const std::pair<int, size_t>& lhs, const std::pair<int, size_t>& rhs){
        if(lhs.second == rhs.second){
            return lhs.first < rhs.first;
        }
        return lhs.second > rhs.second;
    };
    std::vector<std::pair<uint64_t, size_t>> ktop_rows = top_rows.gather_topk(k, comp_count);

    if(world.rank0()){
        for(auto &p : ktop_rows){
            printf("row: %d, count: %d\n", p.first, p.second);
        }
    }
}