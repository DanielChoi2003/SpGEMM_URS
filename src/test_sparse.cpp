#include "sorted_coo.hpp"
#include "rmat_graph_generator/rmat_graph_generator.hpp"
#include <ygm/container/bag.hpp>
#include <ygm/io/csv_parser.hpp>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <boost/container_hash/hash.hpp>

/*
    MEM REQUIREMENT:
        scale 19 (100% RMAT) edges approx: 5,042,946,712
        Each edge contains: 3 uint64_t for row, column, value; 3 * 8 bytes = 24 bytes per edge
        for matrix C: 5,042,946,712 * 24 bytes = 121,030,721,088 = 121 GB?
        for matrix B: 8,388,608 edges
                      8,388,608 * 24 bytes = 201,326,592 bytes = 200 MB
        for matrix A: ~200 MB
*/

/*
    scale = 16
    with -g and -pg, run one with cache and one without cache
    -Ofast/O3 inline functions, making gprof output's time measurement hard to interpret

    data scaling experiment
    16 nodes, each node having number of processors that is whatever is the fastest 
    increase from 16 to 25.
    run with cache and no cache.

    boost unordered flat set count() vs contains()

    Bloom filter: replacing the two unordered flat sets

    prepopulating the cache: moves the rehashing to the preprocessing part, preventing it from rehashing during multiplication
*/

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
        std::string rmat_17 = "/g/g14/choi26/SpGEMM_Project2/data/scale_17.csv";

        std::string amazon_output = "/usr/workspace/choi26/data/real_results/amazon_numpy_output.csv";
        std::string epinions_output = "/g/g14/choi26/graphBLAS_sandbox/graphblas_epinions_result.csv";

        std::string filename_A = rmat_17;
        std::string filename_B = rmat_17;

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

        //#define MAKE_COPY
        std::string rmat_17_output = "/usr/workspace/choi26/scale_17_result.csv";

        if(argc != 2){
            if(world.rank0()){
                std::cerr << "Error: Not enough arguments provided." << std::endl;
                std::cerr << "Usage: " << argv[0] << " <scale value>" << std::endl;
            }
            return 1;
        }
        
        int scale = std::atoi(argv[1]);
        if(scale <= 16){
            if(world.rank0()){
                std::cerr << "RMAT Hashing function does not accept scale value below or equal to 16." << std::endl;
            }
            return 1;
        }

        world.cout0("scale: ", scale);
        int edge_factor = 16;
        int edges = pow(2, scale) * edge_factor;
        double a = 0.57;
        double b = 0.19;
        double c = 0.19;
        double d = 0.05;
        double rmat_to_uni_ratio = 1;

        ygm::container::array<Edge> unsorted_matrix(world, edges);
        ygm::container::array<Edge> sorted_matrix(world, edges);

        auto top_row_ptr = std::make_unique<ygm::container::counting_set<uint64_t>>(world);
        auto top_col_ptr = std::make_unique<ygm::container::counting_set<uint64_t>>(world);
        auto top_row_ygm_ptr = world.make_ygm_ptr(*top_row_ptr);
        auto top_col_ygm_ptr = world.make_ygm_ptr(*top_col_ptr);
        rmat_graph_generator rmat_gen_A(world, unsorted_matrix, top_row_ygm_ptr);
        rmat_gen_A.generate_rmat_edges(scale, edges, a, b, c, d, rmat_to_uni_ratio, true, false, false);

        #ifdef MAKE_COPY
            std::stringstream filename;
            filename << "/usr/workspace/choi26/output_rank_" << world.rank() << ".csv";
            std::ofstream out(filename.str());

            // for_all calls barrier, whereas local_for_all does not
            unsorted_matrix.for_all([&](const auto& index, const auto& edge) {
                out << edge.row << "," << edge.col << "\n";
            });
            out.close();
            world.barrier();
            if (world.rank() == 0) {
                std::stringstream merged_filename;
                merged_filename << "/usr/workspace/choi26/uni_scale_" << std::to_string(scale) << ".csv";
                std::ofstream merged(merged_filename.str());
                
                for (int r = 0; r < world.size(); ++r) {
                    std::stringstream rank_file;
                    rank_file << "/usr/workspace/choi26/output_rank_" << r << ".csv";
                    std::ifstream in(rank_file.str());
                    merged << in.rdbuf();  // Fast copy
                    in.close();
                    std::filesystem::remove(rank_file.str());  // Clean up
                }
            }
            return 0;
        #endif
    
        rmat_graph_generator rmat_gen_B(world, sorted_matrix, top_col_ygm_ptr);
        rmat_gen_B.generate_rmat_edges(scale, edges, a, b, c, d, rmat_to_uni_ratio, false, true, true);

        world.barrier();
        // NOTE: YGM::BAG'S CLEAR() DOES NOT DEALLOCATE THE MEMORY/CAPACITY
    #endif

    double setup_start = MPI_Wtime();
    size_t k = 1000;
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

    #define FILTER
    #ifdef FILTER
        // FILTERING TOP ROWS AND COLUMNS
        std::unordered_set<uint64_t> top;
        for(auto &p : ktop_rows){
            top.insert(p.first);
        }
        world.cout0("before size: ", unsorted_matrix.size());
        auto bagap = std::make_unique<ygm::container::bag<Edge>>(world);
        auto bagbp = std::make_unique<ygm::container::bag<Edge>>(world);
        unsorted_matrix.for_all([&](int index, Edge &ed){
            if(!top.contains(ed.row)){
                bagap->async_insert(ed);
            }
        });
        sorted_matrix.for_all([&](int index, Edge &ed){
            if(!top.contains(ed.col)){
                bagbp->async_insert(ed);
            }
        });
        world.barrier();
        world.cout0("filtered size: ", bagap->size());
        ygm::container::array<Edge> filtered_unsorted_matrix(world, *bagap);
        bagap.reset();
        ygm::container::array<Edge> filtered_sorted_matrix(world, *bagbp);
        bagbp.reset();
    #endif

    #if defined(FILTER)
        Sorted_COO test_COO(world, filtered_sorted_matrix, k, ktop_rows, ktop_cols);
    #else
        Sorted_COO test_COO(world, sorted_matrix, k, ktop_rows, ktop_cols);
    #endif
    double setup_end = MPI_Wtime();
    world.cout0("setup time: ", setup_end - setup_start);

    ygm::container::map<map_key, sum_counter> matrix_C(world); 
    double spgemm_start = MPI_Wtime();
    #if defined(FILTER)
        test_COO.spGemm(filtered_unsorted_matrix, matrix_C);
    #else
        test_COO.spGemm(unsorted_matrix, matrix_C);
    #endif
    world.barrier();
    double spgemm_end = MPI_Wtime();    
    world.cout0("Total number of cores: ", world.size());
    world.cout0("matrix multiplication time: ", spgemm_end - spgemm_start);

    world.cout0(matrix_C.size());
    // MEASURE PUSHES INTO MATRIX C
    // auto counter_comp = [](auto const &a, auto const &b){
    //     if(a.second.push == b.second.push){
    //         return a.second.sum > b.second.sum;
    //     }
    //     return a.second.push > b.second.push;
    // };
    // auto top_k = matrix_C.gather_topk(10, counter_comp);

    // world.barrier();
    // if(world.rank0()){
    //     for(auto &top_entry : top_k){
    //         printf("(%d, %d) had push counter of %d\n", top_entry.first.x, top_entry.first.y, top_entry.second.push);
    //     }
    // }
   

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

        //#define CSV_COMPARE
        #ifdef CSV_COMPARE
        std::string output = "./output.csv";
        std::string expected_output = rmat_17_output;

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