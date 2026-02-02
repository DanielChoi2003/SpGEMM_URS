#include "sorted_coo.hpp"
#include "rmat_graph_generator/rmat_graph_generator.hpp"
#include <ygm/container/bag.hpp>
#include <ygm/io/csv_parser.hpp>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <boost/container_hash/hash.hpp>


int main(int argc, char** argv){

    ygm::comm world(&argc, &argv);
    static ygm::comm &s_world = world;
    
    //#define CSV
    #define RMAT

    #ifdef CSV
        //#define UNDIRECTED_GRAPH
        // uncomment this if you want a AA multiplication but A is not a square matrix
        #define TRANSPOSE 

        std::string livejournal =  "/usr/workspace/choi26/com-lj.ungraph.csv";
        std::string amazon = "/usr/workspace/choi26/data/real_data/undirected_single_edge/com-amazon.ungraph.csv";
        std::string epinions = "/usr/workspace/choi26/data/real_data/directed/soc-Epinions1.csv";

        std::string amazon_output = "/usr/workspace/choi26/data/real_results/amazon_numpy_output.csv";
        std::string epinions_output = "/g/g14/choi26/graphBLAS_sandbox/graphblas_epinions_result.csv";

        std::string filename_A = livejournal;
        std::string filename_B = livejournal;

        // Task 1: data extraction
        auto bagap = std::make_unique<ygm::container::bag<Edge>>(world);
        auto top_row_ptr = std::make_unique<ygm::container::counting_set<uint64_t>>(world);
        std::vector<std::string> files_A= {filename_A};
        std::fstream file_A(files_A[0]);
        YGM_ASSERT_RELEASE(file_A.is_open() == true);
        file_A.close();
        ygm::io::csv_parser parser_A(world, files_A);
        // if the data is small, only one rank will participate
        parser_A.for_all([&](ygm::io::detail::csv_line line){ 

            uint64_t row = line[0].as_integer();
            uint64_t col = line[1].as_integer();
            uint64_t value = 1;
            if(line.size() == 3){
            value = line[2].as_integer();
            }
            // what about self directed edge?
            #ifdef UNDIRECTED_GRAPH
                Edge rev = {col, row, value};
                bagap->async_insert(rev);
                top_row_ptr->async_insert(col);
            #endif
            Edge ed = {row, col, value};
            bagap->async_insert(ed);
            top_row_ptr->async_insert(row);
        });
        world.barrier();

        ygm::container::array<Edge> unsorted_matrix(world, *bagap);
        bagap.reset();

        // matrix B data extraction
        auto bagbp = std::make_unique<ygm::container::bag<Edge>>(world);
        auto top_col_ptr = std::make_unique<ygm::container::counting_set<uint64_t>>(world);
        std::vector<std::string> files_B= {filename_B};
        std::fstream file_B(files_B[0]);
        YGM_ASSERT_RELEASE(file_B.is_open() == true);
        file_B.close();
        ygm::io::csv_parser parser_B(world, files_B);
        parser_B.for_all([&](ygm::io::detail::csv_line line){

            uint64_t row = line[0].as_integer();
            uint64_t col = line[1].as_integer();
            uint64_t value = 1;
            if(line.size() == 3){
                value = line[2].as_integer();
            }
            #if defined(UNDIRECTED_GRAPH) || defined(TRANSPOSE)
                Edge rev = {col, row, value};
                bagbp->async_insert(rev);
                top_col_ptr->async_insert(row);
            #endif


            #ifndef TRANSPOSE
                Edge ed = {row, col, value};
                bagbp->async_insert(ed);
                top_col_ptr->async_insert(col);
            #endif
        });
        world.barrier();

        ygm::container::array<Edge> sorted_matrix(world, *bagbp);
        bagbp.reset();

    #endif

    #ifdef RMAT
        auto unsorted_bag_ptr = std::make_unique<ygm::container::bag<Edge>>(world);
        auto sorted_bag_ptr = std::make_unique<ygm::container::bag<Edge>>(world);
        auto top_row_ptr = std::make_unique<ygm::container::counting_set<uint64_t>>(world);
        auto top_col_ptr = std::make_unique<ygm::container::counting_set<uint64_t>>(world);
        auto top_row_ygm_ptr = world.make_ygm_ptr(*top_row_ptr);
        auto top_col_ygm_ptr = world.make_ygm_ptr(*top_col_ptr);

        rmat_graph_generator rmat_gen_A(world, *unsorted_bag_ptr, top_row_ygm_ptr);
        rmat_graph_generator rmat_gen_B(world, *sorted_bag_ptr, top_col_ygm_ptr);

        int scale = 16;
        int edge_factor = 10;
        int edges = pow(2, scale) * edge_factor;
        double a = 0.57;
        double b = 0.19;
        double c = 0.19;
        double d = 0.05;
        double rmat_to_uni_ratio = 1.0;
        rmat_gen_A.generate_rmat_edges(scale, edges, a, b, c, d, rmat_to_uni_ratio, true, false, false);
        rmat_gen_B.generate_rmat_edges(scale, edges, a, b, c, d, rmat_to_uni_ratio, false, true, true);

        world.barrier();

        ygm::container::array<Edge> unsorted_matrix(world, *unsorted_bag_ptr);
        ygm::container::array<Edge> sorted_matrix(world, *sorted_bag_ptr);
        // NOTE: YGM::BAG'S CLEAR() DOES NOT DEALLOCATE THE MEMORY/CAPACITY
        unsorted_bag_ptr.reset();
        sorted_bag_ptr.reset();
    #endif

    double setup_start = MPI_Wtime();
    size_t k = 100;
    auto comp_count = [](const std::pair<uint64_t, size_t>& lhs, const std::pair<uint64_t, size_t>& rhs){
        if(lhs.second == rhs.second){
            return lhs.first < rhs.first;
        }
        return lhs.second > rhs.second;
    };
    std::vector<std::pair<uint64_t, size_t>> ktop_cols = (*top_col_ptr).gather_topk(k, comp_count);
    std::vector<std::pair<uint64_t, size_t>> ktop_rows = (*top_row_ptr).gather_topk(k, comp_count);
    top_row_ptr.reset();
    top_col_ptr.reset();
    world.barrier();
    Sorted_COO test_COO(world, sorted_matrix, k, ktop_rows, ktop_cols);
    double setup_end = MPI_Wtime();
    world.cout0("setup time: ", setup_end - setup_start);

    ygm::container::map<map_key, sum_counter> matrix_C(world); 
    double spgemm_start = MPI_Wtime();
    test_COO.spGemm(unsorted_matrix, matrix_C);
    world.barrier();
    double spgemm_end = MPI_Wtime();    
    world.cout0("Total number of cores: ", world.size());
    world.cout0("matrix multiplication time: ", spgemm_end - spgemm_start);

    world.cout0(matrix_C.size());
    auto counter_comp = [](auto const &a, auto const &b){
        if(a.second.push == b.second.push){
            return a.second.sum > b.second.sum;
        }
        return a.second.push > b.second.push;
    };
    auto top_k = matrix_C.gather_topk(10, counter_comp);

    world.barrier();
    if(world.rank0()){
        for(auto &top_entry : top_k){
            printf("(%d, %d) had push counter of %d\n", top_entry.first.x, top_entry.first.y, top_entry.second.push);
        }
    }
   

    //#define MATRIX_OUTPUT
    #ifdef MATRIX_OUTPUT
   
    ygm::container::bag<Edge> global_bag_C(world);
    matrix_C.for_all([&global_bag_C](map_key coord, sum_counter sc){
        global_bag_C.async_insert({coord.x, coord.y, sc.sum});
    });
    world.barrier();

    std::vector<Edge> sorted_output_C;
    global_bag_C.gather(sorted_output_C, 0);
    if(world.rank0()){
        std::ofstream output_file;
        output_file.open("./output.csv");
        std::sort(sorted_output_C.begin(), sorted_output_C.end());
        for(Edge &ed : sorted_output_C){
            output_file << ed.row << "," << ed.col << "," << ed.value << "\n";
            //printf("%d,%d,%d\n", ed.row, ed.col, ed.value);
        }
        output_file.close();

        #define CSV_COMPARE
        #ifdef CSV_COMPARE
        std::string output = "./output.csv";
        std::string expected_output = epinions_output;

        //"../strong_scaling_output/epinions_results/second_epinions_strong_scaling_${i}_nodes.txt"
        // ignore all: > /dev/null 2>&1

        int nodes = world.size() / 32;
        std::string cmd = "diff -y --suppress-common-lines "
                        + output + " " + expected_output + 
                        " > ../strong_scaling_output/epinions_results/" +
                        std::to_string(nodes) + "_nodes_difference.txt";

        int result = system(cmd.c_str());

        std::filesystem::remove("./output.csv");
        if (result == 0) {
            std::cout << "Files match!\n";
            std::filesystem::remove(
                        "../strong_scaling_output/epinions_results/" +
                        std::to_string(nodes) + 
                        "_nodes_difference.txt"
                    );
        } else {
            std::cout << "Files differ!\n";
        }
        #endif
    }
    #endif

    return 0;
}