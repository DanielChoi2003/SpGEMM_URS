#include "sorted_coo.hpp"
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
    
    //#define UNDIRECTED_GRAPH

    std::string livejournal =  "/usr/workspace/choi26/com-lj.ungraph.csv";
    std::string amazon = "../data/real_data/undirected_single_edge/com-amazon.ungraph.csv";
    std::string epinions = "../data/real_data/directed/soc-Epinions1.csv";

    std::string amazon_output = "../data/real_results/amazon_numpy_output.csv";
    std::string epinions_output = "../data/real_results/Epinions_numpy_output.csv";

    std::string filename_A = livejournal;
    std::string filename_B = livejournal;

     // Task 1: data extraction
    auto bagap = std::make_unique<ygm::container::bag<Edge>>(world);
    ygm::container::counting_set<int> top_cols(world);
    std::vector<std::string> files_A= {filename_A};
    std::fstream file_A(files_A[0]);
    YGM_ASSERT_RELEASE(file_A.is_open() == true);
    file_A.close();
    ygm::io::csv_parser parser_A(world, files_A);
    // if the data is small, only one rank will participate
    parser_A.for_all([&](ygm::io::detail::csv_line line){ 

        int row = line[0].as_integer();
        int col = line[1].as_integer();
        int value = 1;
        if(line.size() == 3){
           value = line[2].as_integer();
        }
        #ifdef UNDIRECTED_GRAPH
            Edge rev = {col, row, value};
            bagap->async_insert(rev);
            top_cols.async_insert(col);
        #endif
        Edge ed = {row, col, value};
        bagap->async_insert(ed);
        top_cols.async_insert(row);
    });
    world.barrier();

    ygm::container::array<Edge> unsorted_matrix(world, *bagap);
    bagap.reset();

    // matrix B data extraction
    auto bagbp = std::make_unique<ygm::container::bag<Edge>>(world);
    ygm::container::counting_set<int> top_rows(world);
    std::vector<std::string> files_B= {filename_B};
    std::fstream file_B(files_B[0]);
    YGM_ASSERT_RELEASE(file_B.is_open() == true);
    file_B.close();
    ygm::io::csv_parser parser_B(world, files_B);
    parser_B.for_all([&](ygm::io::detail::csv_line line){

        int row = line[0].as_integer();
        int col = line[1].as_integer();
        int value = 1;
        if(line.size() == 3){
            value = line[2].as_integer();
        }
        #ifdef UNDIRECTED_GRAPH
            Edge rev = {col, row, value};
            bagbp->async_insert(rev);
            top_rows.async_insert(row);
        #endif
        Edge ed = {row, col, value};
        bagbp->async_insert(ed);
        top_rows.async_insert(col);
    });
    world.barrier();

    ygm::container::array<Edge> sorted_matrix(world, *bagbp);
    bagbp.reset();

    double setup_start = MPI_Wtime();
    size_t k = 100;
    auto comp_count = [](const std::pair<int, size_t>& lhs, const std::pair<int, size_t>& rhs){
        if(lhs.second == rhs.second){
            return lhs.first < rhs.first;
        }
        return lhs.second > rhs.second;
    };
    std::vector<std::pair<int, size_t>> ktop_rows = top_rows.gather_topk(k, comp_count);
    std::vector<std::pair<int, size_t>> ktop_cols = top_cols.gather_topk(k, comp_count);
    world.barrier();
    Sorted_COO test_COO(world, sorted_matrix, k, ktop_rows, ktop_cols);
    double setup_end = MPI_Wtime();
    world.cout0("setup time: ", setup_end - setup_start);

    ygm::container::map<map_key, int> matrix_C(world); 
    double spgemm_start = MPI_Wtime();
    test_COO.spGemm(unsorted_matrix, matrix_C);
    world.barrier();
    double spgemm_end = MPI_Wtime();    
    world.cout0("Total number of cores: ", world.size());
    world.cout0("matrix multiplication time: ", spgemm_end - spgemm_start);

    //#define MATRIX_OUTPUT
    #ifdef MATRIX_OUTPUT
   
    ygm::container::bag<Edge> global_bag_C(world);
    matrix_C.for_all([&global_bag_C](map_key coord, int product){
        global_bag_C.async_insert({coord.x, coord.y, product});
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
        std::string expected_output = amazon_output;

        //"../strong_scaling_output/epinions_results/second_epinions_strong_scaling_${i}_nodes.txt"
        // ignore all: > /dev/null 2>&1

        int nodes = world.size() / 32;
        std::string cmd = "diff -y --suppress-common-lines "
                        + output + " " + expected_output + 
                        " > ../strong_scaling_output/epinions_results/" +
                        std::to_string(nodes) + "_nodes_difference.txt"; // TESTING

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