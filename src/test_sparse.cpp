#include "sorted_coo.hpp"
#include <ygm/io/csv_parser.hpp>
#include <stdio.h>
#include <mpi.h>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <boost/container_hash/hash.hpp>


struct map_key{
    int x;
    int y;

    bool operator==(const map_key& other) const {
        return x == other.x && y == other.y;
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(x, y);
    }
};

/*
    std::pair is not trivially copyable -> need to use struct ->
    requires custom hashing for the struct as std::pair is no longer
    used
*/
std::size_t hash_value(map_key const& key) {
  std::size_t seed = 0;
  boost::hash_combine(seed, key.x);
  boost::hash_combine(seed, key.y);
  return seed;
}

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
        #endif
        Edge ed = {row, col, value};
        bagap->async_insert(ed);
    });
    world.barrier();

    ygm::container::array<Edge> unsorted_matrix(world, *bagap);
    bagap.reset();

    // matrix B data extraction
    auto bagbp = std::make_unique<ygm::container::bag<Edge>>(world);
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
        #endif
        Edge ed = {row, col, value};
        bagbp->async_insert(ed);
    });
    world.barrier();

    ygm::container::array<Edge> sorted_matrix(world, *bagbp);
    bagbp.reset();

    double setup_start = MPI_Wtime();
    Sorted_COO test_COO(world, sorted_matrix);
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
        std::string expected_output = epinions_output;

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


    //#define TRIANGLE_COUNTING
    #ifdef TRIANGLE_COUNTING
    double bag_C_start = MPI_Wtime();
    auto bagcp = std::make_unique<ygm::container::bag<Edge>>(world);
    matrix_C.for_all([&bagcp](std::pair<int, int> indices, int value){
        bagcp->async_insert({indices.first, indices.second, value});
    });
    world.barrier();
    double bag_C_end = MPI_Wtime();
    world.cout0("Constructing bag C from map matrix C took ", bag_C_end - bag_C_start, " seconds");
    ygm::container::array<Edge> arr_matrix_C(world, *bagcp);  // <row, col>, partial product 
    bagcp.reset();

    ygm::container::map<std::pair<int, int>, int> diagonal_matrix(world);  //

    double triangle_count_start = MPI_Wtime();
    test_COO.spGemm(arr_matrix_C, diagonal_matrix);
    world.barrier();

    // diagonal_matrix.for_all([](std::pair<int, int> pair, int product){
    //     if(pair.first == pair.second){
    //         printf("%d, %d, %d\n", pair.first, pair.second, product);
    //     }
    // });

    int triangle_count = 0;
    int global_triangle_count = 0;
    auto global_triangle_ptr = world.make_ygm_ptr(global_triangle_count);
    diagonal_matrix.for_all([&triangle_count](std::pair<int, int> indices, int value){
        if(indices.first == indices.second){
            triangle_count += value;
        }
    });
    world.barrier();

    auto adder = [](int value, auto global_triangle_ptr){
        *global_triangle_ptr += value;
    };
    world.async(0, adder, triangle_count, global_triangle_ptr);
    world.barrier();

    double triangle_count_end = MPI_Wtime();
    world.cout0("Triangle counting and convergence took ", triangle_count_end - triangle_count_start, " seconds");
    if(world.rank0()){
        s_world.cout0("triangle count: ", global_triangle_count, " / 6 = ", global_triangle_count / 6);
    }
    #endif

   

    return 0;
}