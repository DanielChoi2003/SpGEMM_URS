#include "sorted_coo.hpp"
#include "rmat_graph_generator/rmat_graph_generator.hpp"
#include "shm_hub/shm_hub.h"
#include <ygm/io/csv_parser.hpp>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <ctime>
#include <string>
#include <filesystem>
#include <algorithm>
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

/* 3/16/2026
 * hard-coded threshold for determining the hubs
 * replicating the hub edges 
 * use a set to determine if its a hub edge or not
 * statistics for max and average (use ygm::max, ygm::sum) DONE
 * Setting a memory threshold for replicated hub edges?
*/
struct Config{
    bool enableRMAT = false;
    int scale = -1;

    bool enableCSV = false;
    std::string CSVInput1 = "";
    std::string CSVInput2 = "";

    bool enableUndirected = false;

    bool enableTranspose = false;

    bool enableCopy = false;

    bool enableOutput = false;
    std::string outputFile = "";

    bool enableCSVCompare = false; 
    std::string outputCompare = "";
};

bool checkNumber(std::string s){
    for(int i = 0; i < s.size(); i++){
        if(!isdigit(s[i])){
            return false;
        }
    }
    return true;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n\n"
              << "Options:\n"
              << "  -r, --rmat <value>       Set the scale value\n"
              << "  --csv <file1> <file2>    Enable CSV input\n"
              << "  -c, --copy               Enable RMAT input copy\n"
              << "  --compare <file>         Compare output with a given file\n"
              << "  -o, --output <file>      Set output file\n"
              << "  -t, --transpose          Enable edge transpose\n"
              << "  -u, --undirected         Enable undirected edges\n"
              << "  -h, --help               Show this help message\n\n"
              << "Example:\n"
              << "  " << programName << " -r 18\n";
}

Config parseArgs(int argc, char* argv[], ygm::comm &world) {
    Config config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            if(world.rank0()){
                printUsage(argv[0]);
            }
            exit(0); 
        } else if ((arg == "-r" || arg == "--rmat") && i + 1 < argc) {
            config.enableRMAT = true;
            config.enableCSV = false;
            std::string num = argv[++i];
            if(!checkNumber(num)){
                if(world.rank0()){
                    std::cerr << "Error: Non-integer argument provided." << std::endl;
                    std::cerr << "Usage: " << argv[0] << " <scale value>" << std::endl;
                }
                exit(1);
            }
            config.scale = std::stoi(num);
        } else if (arg == "--csv" && i + 2 < argc) {
            config.enableCSV = true;
            config.enableRMAT = false;
            config.CSVInput1 = argv[++i];
            config.CSVInput2 = argv[++i];
        } else if (arg == "-t" || arg == "--transpose") {
            config.enableTranspose = true;
        } else if (arg == "-u" || arg == "--undirected") {
            config.enableUndirected = true;
        } else if ((arg == "-c" || arg == "--copy") && i + 1 <= argc) {
            config.enableCopy = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 <= argc) {
            config.enableOutput = true;
            config.outputFile = argv[++i];
        } else if (arg == "--compare" && i + 1 < argc) {
            config.enableCSVCompare = true;
            config.outputCompare = argv[++i];
        }else {
            if(world.rank0()){
                std::cout << "Unknown argument: " << arg << "\n";
                std::cout << "Run with -h for help.\n";
            }
            exit(1);
        }
    }

    return config;
}

/*
    some example inputs and outputs:
    
        std::string livejournal =  "/usr/workspace/choi26/com-lj.ungraph.csv";
        std::string amazon = "/usr/workspace/choi26/data/real_data/undirected_single_edge/com-amazon.ungraph.csv";
        std::string epinions = "/usr/workspace/choi26/data/real_data/directed/soc-Epinions1.csv";
        std::string rmat_17 = "/g/g14/choi26/SpGEMM_Project2/data/scale_17.csv";

        std::string amazon_output = "/usr/workspace/choi26/data/real_results/amazon_numpy_output.csv";
        std::string epinions_output = "/g/g14/choi26/graphBLAS_sandbox/graphblas_epinions_result.csv";
*/

int main(int argc, char** argv){

    ygm::comm world(&argc, &argv);
    static ygm::comm &s_world = world;
    Config config = parseArgs(argc, argv, world);
    
    world.welcome();

    std::unique_ptr<ygm::container::array<Edge>> sorted_matrix;
    std::unique_ptr<ygm::container::array<Edge>> unsorted_matrix;
    auto A_column_degree = std::make_unique<ygm::container::counting_set<uint64_t>>(world);
    auto B_row_degree = std::make_unique<ygm::container::counting_set<uint64_t>>(world);
    auto bagbp = std::make_unique<ygm::container::bag<Edge>>(world);
    double A_deg_avg, B_deg_avg; 
    if(config.enableCSV){

        std::string filename_A = config.CSVInput1;
        std::string filename_B = config.CSVInput2;

        auto bagap = std::make_unique<ygm::container::bag<Edge>>(world);
        auto top_row_ptr = std::make_unique<ygm::container::counting_set<uint64_t>>(world);
        std::vector<std::string> files_A = {filename_A};
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
            if(config.enableUndirected){
                Edge rev = {col, row, value};
                bagap->async_insert(rev);
                A_column_degree->async_insert(row);
            }
            Edge ed = {row, col, value};
            bagap->async_insert(ed);
            top_row_ptr->async_insert(row);
            A_column_degree->async_insert(col);
        });
        world.barrier();

        unsorted_matrix = std::make_unique<ygm::container::array<Edge>>(world, *bagap);
        bagap.reset();

        // matrix B data extraction
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
            if(config.enableTranspose || config.enableUndirected){
                Edge rev = {col, row, value};
                bagbp->async_insert(rev);
                B_row_degree->async_insert(col);
            } else{
                Edge ed = {row, col, value};
                bagbp->async_insert(ed);
                B_row_degree->async_insert(row);
            }
        });
        world.barrier();

        sorted_matrix = std::make_unique<ygm::container::array<Edge>>(world, *bagbp);
        //bagbp.reset();
    } 
    else if(config.enableRMAT){
        // currently RMAT does NOT work
        if(world.rank0()){
            std::cerr << "RMAT generator does not work properly. Please use CSV option instead." << std::endl;
        }
        return 1;
        if(config.scale <= 16){
            if(world.rank0()){
                std::cerr << "RMAT Hashing function does not accept scale value below or equal to 16." << std::endl;
            }
            return 1;
        }

        world.cout0("scale: ", config.scale);
        int edge_factor = 16;
        int edges = pow(2, config.scale) * edge_factor;
        double a = 0.57;
        double b = 0.19;
        double c = 0.19;
        double d = 0.05;
        double rmat_to_uni_ratio = 1;

        unsorted_matrix = std::make_unique<ygm::container::array<Edge>>(world, edges);
        sorted_matrix = std::make_unique<ygm::container::array<Edge>>(world, edges);

        rmat_graph_generator rmat_gen_A(world, *unsorted_matrix);
        rmat_gen_A.generate_rmat_edges(config.scale, edges, a, b, c, d, rmat_to_uni_ratio, false, false, false);

        if(config.enableCopy){
            std::stringstream filename;
            filename << "/usr/workspace/choi26/output_rank_" << world.rank() << ".csv";
            std::ofstream out(filename.str());

            (*unsorted_matrix).for_all([&](const auto& index, const auto& edge) {
                out << edge.row << "," << edge.col << "\n";
            });
            out.close();
            world.barrier();
            if (world.rank() == 0) {
                std::stringstream merged_filename;
                merged_filename << "/usr/workspace/choi26/rmat_scale_" << std::to_string(config.scale) << ".csv";
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
        }
    
        rmat_graph_generator rmat_gen_B(world, *sorted_matrix);
        rmat_gen_B.generate_rmat_edges(config.scale, edges, a, b, c, d, rmat_to_uni_ratio, false, false, true);

        world.barrier();
        // NOTE: YGM::BAG'S CLEAR() DOES NOT DEALLOCATE THE MEMORY/CAPACITY
    }
    
    static size_t topk = 1000;
    size_t max_hub_edges = 1024*1024; // NOT A STRICT CAP; MAY EXCEED
    std::unique_ptr<ygm::container::array<Edge>> nonhub_edges;
    shm_hub<Edge> SHM_HUB(world, topk, max_hub_edges, B_row_degree, nonhub_edges, bagbp);

    return 0;
    // Replace HUB_EDGES and B_HUBS with SHM_HUB object!
    // Sorted_COO test_COO(world, nonhub_edges, hub_edges, B_hubs);
    // ygm::container::map<map_key, uint64_t> matrix_C(world); 
    // double spgemm_start = MPI_Wtime();
    // test_COO.spGemm(*unsorted_matrix, matrix_C);
    // world.barrier();
    // double spgemm_end = MPI_Wtime();    
    // world.cout0("Total number of cores: ", world.size());
    // world.cout0("matrix multiplication time: ", spgemm_end - spgemm_start);

    // world.cout0("matrix C size: ", matrix_C.size());   

    // if(config.enableOutput){
    //     ygm::container::bag<Edge> global_bag_C(world);
    //     matrix_C.for_all([&global_bag_C](map_key coord, auto value){
    //         global_bag_C.async_insert({coord.x, coord.y, value});
    //     });
    //     world.barrier();

    //     std::vector<Edge> sorted_output_C;
    //     global_bag_C.gather(sorted_output_C, 0);
    //     if(world.rank0()){
    //         std::ofstream output_file;
    //         output_file.open(config.outputFile);
    //         std::sort(sorted_output_C.begin(), sorted_output_C.end());
    //         for(Edge &ed : sorted_output_C){
    //             output_file << ed.row << "," << ed.col << "," << ed.value << "\n";
    //         }
    //         output_file.close();    
    //     }
    // }
    
    // if(config.enableCSVCompare){
    //     /*
    //      * If output file has already been created, use it.
    //      * else, create a temporary output file
    //     */
    //     std::string output = "output.csv";
    //     std::string expected_output = config.outputCompare;

    //     if(config.enableOutput){
    //         output = config.outputFile;
    //     }
    //     else{
    //         ygm::container::bag<Edge> global_bag_C(world);
    //         matrix_C.for_all([&global_bag_C](map_key coord, auto value){
    //             global_bag_C.async_insert({coord.x, coord.y, value});
    //         });
    //         world.barrier();

    //         std::vector<Edge> sorted_output_C;
    //         global_bag_C.gather(sorted_output_C, 0);
    //         if(world.rank0()){
    //             std::ofstream output_file;
    //             output_file.open(output);
    //             std::sort(sorted_output_C.begin(), sorted_output_C.end());
    //             for(Edge &ed : sorted_output_C){
    //                 output_file << ed.row << "," << ed.col << "," << ed.value << "\n";
    //             }
    //             output_file.close();
    //         }
    //     }

    //     if(world.rank0()){
    //         // ignore all: > /dev/null 2>&1
    //         pid_t pid = getpid();
    //         time_t now = time(0);
    //         std::string unique_id = std::to_string(pid) + std::to_string(now);
    //         std::string cmd = "diff -y --suppress-common-lines "
    //                         + output + " " + expected_output + 
    //                         " > " + unique_id + "_nodes_difference.txt";
    //         int result = system(cmd.c_str());

    //         std::filesystem::remove("./output.csv");
    //         if (result == 0) {
    //             std::cout << "Files match!\n";
    //             std::filesystem::remove(
    //                         "../strong_scaling_output/epinions_results/" +
    //                         unique_id + 
    //                         "_nodes_difference.txt"
    //                     );
    //         } else {
    //             std::cout << "Files differ!\n";
    //         }
    //     }
    // }
   
    

    return 0;
}