// Copyright 2019-2025 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <boost/unordered/unordered_flat_set.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <ygm-gctc/kronecker_edge_generator_testing.hpp>
#include <ygm/comm.hpp>
#include <ygm/container/counting_set.hpp>
#include <ygm/container/map.hpp>
#include <set>

  // IMPORTANT: need to make triangle_neighbor a set due to duplicates
    // OR, filter it later with unordered set. but what is more efficient?
struct vert_info{
    template <class Archive>
    void serialize( Archive & ar )
    {
      ar(triangle_count, core_count, noncore_count, total_count, triangle_neighbor, triangle_centrality);
    }
      uint64_t triangle_count = 0;
      uint64_t core_count = 0;
      uint64_t noncore_count = 0;
      uint64_t total_count = 0;
      std::vector<uint64_t> triangle_neighbor;
      double triangle_centrality = 0;
};


using graph_type = ygm::container::map<int, vert_info >;


// finds the appropriate graph files based on the user input
std::pair<std::string, std::string> parse_cmdline(int argc, char **argv,
                                                  ygm::comm &comm) {

  YGM_ASSERT_RELEASE(argc <= 2);

  int scale = 20; 
  if (argc == 2) {
    scale = atoi(argv[1]);
  }

  if (scale < 20 || scale > 42) {
    comm.cout0("ERROR:  Minimum scale is 20, Maximum scale is 42");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int A_scale = scale / 2;
  int B_scale = scale / 2 + (scale % 2 == 1);

  std::stringstream A_sstr, B_sstr;

  
  A_sstr << "./data/zachary_karate.edges";
  // B_sstr << "./data/B_G500_S_testing_tc.edges";


  comm.cout0("SCALE: ", scale);
  comm.cout0("Input A: ", A_sstr.str());
  comm.cout0("Input B: ", B_sstr.str());

  return {A_sstr.str(), B_sstr.str()};
}

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);

  static ygm::comm& s_world = world;
  {

    world.cout0("YGM Graph Challenge -- Triangle Counting");
    static int s_world_size = world.size();

    auto names = parse_cmdline(argc, argv, world);
    world.welcome();

    gctc::kronecker_edge_generator kron(world, names.first);
    uint64_t global_num_edges = 0;
    uint64_t global_kron_triangles = 0;

    //
    // Count degrees
    ygm::container::counting_set<uint64_t> vertex_degrees(world);
    boost::unordered::unordered_flat_set<uint64_t> set_my_vertices;
    world.cf_barrier(); // what's the difference from a normal barrier()?
    double global_time_start = MPI_Wtime();
    double cd_time_start = MPI_Wtime();
    kron.for_all(
        // this is the fn you pass into the custom for_all()
        [&global_num_edges, &vertex_degrees,
         &set_my_vertices](const auto row, const auto col) { // this uses the newly calculated value of row, col, and val
          if (row < col) { // to avoid duplicates. because graph 1 and 2 are undirected, they will produce a duplicate
            ++global_num_edges;
            vertex_degrees.async_insert(row); // counts how many times a vertex has been inserted
            vertex_degrees.async_insert(col);
            set_my_vertices.insert(row); // stores the rank's owned vertices
            set_my_vertices.insert(col);
          }
    });
    world.barrier();
    // ranks get their own respective vertices mapped to number of occurrences.
    // it only gets a portion of what it owns
    std::map<uint64_t, uint64_t> map_my_vertices = vertex_degrees.gather_keys(
        std::vector<uint64_t>(set_my_vertices.begin(), set_my_vertices.end()));
    world.barrier();
    vertex_degrees.clear();
    set_my_vertices.clear();
    double cd_time_end = MPI_Wtime();
    global_num_edges = ygm::sum(global_num_edges, world);
    world.cout0("global_num_edges = ", global_num_edges);
    world.cout0("map_my_vertices.size() = ", map_my_vertices.size());
    world.cout0("Degree Count Time = ", cd_time_end - cd_time_start);

    
    ygm::container::map<uint64_t, vert_info> vertex_map(world);
    static ygm::container::map<uint64_t, vert_info>& s_vertex_map = vertex_map;
    //
    //  Build dodgr edge list
    struct dodgr_edge {
      uint64_t source;  // local row index
      uint64_t global_source; // actual source vertex ID
      uint64_t dest;
      uint64_t dest_degree;
    } __attribute__((packed)); // tells the compiler to not add byte padding between the data members
    static std::deque<dodgr_edge> dodgr_edges_1d; // degree order directed graph
    world.cf_barrier();
    double rh_time_start = MPI_Wtime();
    kron.for_all([&map_my_vertices, &world](const auto row, const auto col) {
      if (row < col) { 
        YGM_ASSERT_RELEASE(map_my_vertices.count(row) > 0);
        YGM_ASSERT_RELEASE(map_my_vertices.count(col) > 0);
        uint64_t row_degree = map_my_vertices[row];
        uint64_t col_degree = map_my_vertices[col];
        uint64_t source(0);
        uint64_t global_source(0);
        uint64_t target(0);
        uint64_t target_degree(0);

        
        if (row_degree > col_degree) { // the edge redirected towards the higher degree vertex
          source = col;
          target = row;
          target_degree = row_degree;
        } else if (row_degree < col_degree) {
          source = row;
          target = col;
          target_degree = col_degree;
        } else { // equal degree. tie breaker
          if (row > col) {
            source = col;
            target = row;
            target_degree = row_degree;
          } else if (row < col) {
            source = row;
            target = col;
            target_degree = col_degree;
          } else {
            std::cout << "Found a self edge!?!?!" << std::endl;
          }
        }
        //s_world.cout(source, "->", target);

        // PROBLEM: there is a chance that a rank may not push any into its dodgr edge list because source % world.size()
        //          does not result in the rank number
        world.async(
            source % world.size(),
            [](uint64_t source, uint64_t target, uint64_t target_degree) {
              //s_world.cout("pushing source: ", source, " and dest: ", target);
              dodgr_edges_1d.push_back({
                  uint64_t(source / s_world_size),
                  source,
                  target,
                  target_degree
              });
            },
            source, target, target_degree);

        vert_info vi;
        s_vertex_map.async_insert(source, vi);
        s_vertex_map.async_insert(target, vi);       
      }
    });
    world.barrier();
    kron.clear();
    double rh_time_end = MPI_Wtime();
    map_my_vertices.clear();
    std::sort(dodgr_edges_1d.begin(), dodgr_edges_1d.end(),
              [](const auto &ta, const auto &tb) {  // if source is less than, put it first. If equal, compare the dest.
                return std::make_pair(ta.source, ta.dest) <
                       std::make_pair(tb.source, tb.dest);
              });
    static std::vector<size_t> row_jump_index;
    int64_t cur_row = 0;
    row_jump_index.push_back(0);
    for (size_t i = 0; i < dodgr_edges_1d.size(); ++i) {
      size_t local_row_index = dodgr_edges_1d[i].source; // source here is the index
      while (local_row_index > cur_row) { // For any vertex u, where do its edges begin in the edge list?
        ++cur_row;
        row_jump_index.push_back(i);
        YGM_ASSERT_RELEASE(cur_row == row_jump_index.size() - 1);
      }
    }
    //world.cout("local dodgr edge list size: ", dodgr_edges_1d.size());
    static size_t local_largest_vertex = dodgr_edges_1d.back().source; // not actually the largest vertex ID; largest index
    //world.cout("local largest vertex index: ", local_largest_vertex);
    static uint64_t global_source =  dodgr_edges_1d.back().global_source;
    //world.cout("corresponding vertex: ", global_source);
    row_jump_index.push_back(dodgr_edges_1d.size()); // padding to make accessing safe
    row_jump_index.push_back(dodgr_edges_1d.size());
    row_jump_index.push_back(dodgr_edges_1d.size());
    row_jump_index.push_back(dodgr_edges_1d.size());
    YGM_ASSERT_RELEASE(local_largest_vertex < row_jump_index.size());
    static std::vector<size_t> dodgr_edges_1d_targets(dodgr_edges_1d.size());
    for (size_t i = 0; i < dodgr_edges_1d.size(); ++i) {
      dodgr_edges_1d_targets[i] = dodgr_edges_1d[i].dest;
    }
    world.barrier();
    world.cout0("Build dodgr time = ", rh_time_end - rh_time_start);
    world.cout0("Global max local vertex set size = ",
                ygm::max(row_jump_index.size(), world));

    //
    // Generate wedges and check!!
    world.cf_barrier();
    world.stats_reset();
    double wc_time_start = MPI_Wtime();
    size_t global_wedge_checks(0);
    static size_t global_triangles_found(0);
    /*
      original vertex node number: source
      owner of the vertex: source % world.size
      dodgr_edges_1d[i].source = local edge list's index

      so "source" here is simply the index
      "dest" here is the actual destination vertex number
    */
    for (size_t i = 0; i < dodgr_edges_1d.size(); ++i) {
      uint64_t t_i = dodgr_edges_1d[i].source; // local?
      uint64_t global_t_i = dodgr_edges_1d[i].global_source;
      uint64_t t_j = dodgr_edges_1d[i].dest;
      uint64_t t_j_deg = dodgr_edges_1d[i].dest_degree;
      for (size_t j = i + 1; j < dodgr_edges_1d.size(); ++j) {
        if (t_i == dodgr_edges_1d[j].source) { // this is comparing the local edge list index, but if index is equal == the source vertex is equal
          uint64_t t_k = dodgr_edges_1d[j].dest;
          uint64_t t_k_deg = dodgr_edges_1d[j].dest_degree;
          global_wedge_checks++;

          int query_source_rank = 0;
          int query_dest_rank = 0;

          struct packed {
            uint64_t query_source;
            uint64_t query_global_source;
            uint64_t query_target;
          } __attribute__((packed));

          packed p;

          if (t_j_deg < t_k_deg) {  // mark the higher degree vertex as the destination
            p.query_source = t_j / s_world_size;
            p.query_global_source = t_j;
            p.query_target = t_k;
            query_source_rank = t_j % s_world_size;
            query_dest_rank = t_k % s_world_size;
          } else if (t_j_deg > t_k_deg) {
            p.query_source = t_k / s_world_size;
            p.query_global_source = t_k;
            p.query_target = t_j;
            query_source_rank = t_k % s_world_size;
            query_dest_rank = t_j % s_world_size;
          } else if (t_j < t_k) { // if the degree is equal, break the tie with vertex number
            p.query_source = t_j / s_world_size;
             p.query_global_source = t_j;
            p.query_target = t_k;
            query_source_rank = t_j % s_world_size;
            query_dest_rank = t_k % s_world_size;
          } else {
            p.query_source = t_k / s_world_size;
             p.query_global_source = t_k;
            p.query_target = t_j;
            query_source_rank = t_k % s_world_size;
            query_dest_rank = t_j % s_world_size;
          }

          /*  MY EDIT
            Need to increment the other two vertices' local triangle count

            The problem arises when we have to get the sum of total triangle count because originally, it was 
            implemented with an undirected graph. but need to figure a way with the degree order directed graph.
          
            How to get the core count of a vertex? 
            Adding triangle count as triangle is found leads to overcounting. If a neighboring vertex participates in
            two triangles, it will count double

            First, it needs to gather triangle counts for all vertices.
            Second, gather all triangle neighbors.
              Call the neighbors and make the neighbors add to the calling vertex's core count
          */
          world.async(query_source_rank, [p, global_t_i]() { // difference from async_visit?
            size_t local_row_index = p.query_source;
            if (local_row_index <= local_largest_vertex) {
              // lower_bound finds the first occurrence of val or a value that is greater if the exact "val" is not found
              auto itr =
                  std::lower_bound(dodgr_edges_1d_targets.begin() +
                                       row_jump_index[local_row_index],
                                   dodgr_edges_1d_targets.begin() +
                                       row_jump_index[local_row_index + 1],
                                   p.query_target);
              // if it found p.query_target and it matches the destination vertex -> triangle found
              if (itr != dodgr_edges_1d_targets.begin() +
                             row_jump_index[local_row_index + 1] &&
                  *itr == p.query_target) {
                global_triangles_found++; // found a triangle!


                auto updater = [](uint64_t dest, vert_info& vi, uint64_t neighbor1, uint64_t neighbor2){
                  vi.triangle_count++;
                  vi.triangle_neighbor.push_back(neighbor1);
                  vi.triangle_neighbor.push_back(neighbor2);
                };  
                s_vertex_map.async_visit(p.query_global_source, updater, p.query_target, global_t_i);
                s_vertex_map.async_visit(p.query_target, updater, p.query_global_source, global_t_i);
                s_vertex_map.async_visit(global_t_i, updater, p.query_global_source, p.query_target);
              
              }
            }
          });
        } else {
          break; // end of t_i's edges
        }
      }
    }
    world.barrier();
    world.stats_print("TC_TIME");
    double wc_time_end = MPI_Wtime();
    world.barrier();
    global_wedge_checks = ygm::sum(global_wedge_checks, world);
    global_triangles_found = ygm::sum(global_triangles_found, world);
    world.cout0("global_wedge_checks = ", global_wedge_checks);
    world.cout0("global_triangles_found = ", global_triangles_found);
    world.cout0("TC Time = ", wc_time_end - wc_time_start);
    world.cout0("------------------------------------------------------------");
    world.barrier();

    double tc_time_start = MPI_Wtime();


    // local triangle count vector to store triangle counting
    static std::unordered_map<uint64_t, uint64_t> count_map;
    
    // NOTE: might need to run over the query_dest too. If query dest is a high-degree vertex,
    //      then it might not belong in any rank's dodgr_edges_1d as a source. 
    vertex_map.for_all([](int source, vert_info& vi){

      uint64_t triangle_count = vi.triangle_count;
      s_world.async(source % s_world_size, [source, triangle_count](){
        count_map.insert({source, triangle_count});
      });
      vi.core_count += vi.triangle_count;
      std::unordered_set<uint64_t> visited_neighbors; // triangle_neighbors may contain duplicates. this is to filter them out

      for(uint64_t neighbor : vi.triangle_neighbor){
        visited_neighbors.insert(neighbor);
      }

      for(uint64_t neighbor : visited_neighbors){
        auto adder = [source](int dest, vert_info& vi){

          uint64_t dest_count = vi.triangle_count;

          s_vertex_map.async_visit(source, [](int source, vert_info& vi, uint64_t dest_count){
            vi.core_count += dest_count;
          }, dest_count);
        };

        s_vertex_map.async_visit(neighbor, adder);
      }
    });
    world.barrier();

    /*
      The method is to call the dest vertex, add the source vertex's triangle count, 
      and then let it add its triangle count to the source vertex with a nested async
    */
    for(size_t i = 0; i < dodgr_edges_1d.size(); i++){
      uint64_t global_source = dodgr_edges_1d[i].global_source;
      uint64_t dest = dodgr_edges_1d[i].dest;

      uint64_t src_count = count_map.at(global_source);

      vertex_map.async_visit(dest, [src_count](uint64_t dest, vert_info& vi, uint64_t src){

        uint64_t dest_count = vi.triangle_count;
        vi.total_count += src_count;
        
        auto adder = [](uint64_t src, vert_info& vi, uint64_t dest_count){
          vi.total_count += dest_count;

        };

        s_vertex_map.async_visit(src, adder, dest_count);
      }, global_source);

    }
    world.barrier();

    // this might repeat the source vertex. if it has already been processed, then skip
    static double local_max_TC = 0;
    static uint64_t local_vertex;
    vertex_map.for_all([](uint64_t source, vert_info& vi){
      uint64_t total_count = vi.total_count;
      uint64_t core_count = vi.core_count;

      vi.noncore_count = total_count - core_count + vi.triangle_count;
      uint64_t noncore_count = vi.noncore_count;
      //std::cout << "vertex " << global_source << ", core count: " << core_count << ", total count: " << total_count 
                //<< ", noncore_count: " << noncore_count << std::endl;
      vi.triangle_centrality = ((1.0 / 3) * core_count + noncore_count) / global_triangles_found;
      //std::cout << "vertex " << global_source << " has a triangle centrality of " << it->second.triangle_centrality << std::endl;
      if(vi.triangle_centrality > local_max_TC){
        local_max_TC = vi.triangle_centrality;
        local_vertex = source;
      }
      // if(vi.triangle_centrality >= 1){
      //   std::cout << "vertex " << source << ", core count: " << core_count << ", total count: " << total_count 
      //           << ", noncore_count: " << noncore_count << std::endl;
      // }
    });

    world.barrier();
    double global_max_TC = world.all_reduce_max(local_max_TC);
    if(local_max_TC == global_max_TC){
      world.cout("vertex that has max TC: ", local_vertex);
    }
    world.cout0("max triangle centrality: ", global_max_TC);
    world.barrier();
    double tc_time_end = MPI_Wtime();
    world.cout0("TC Time = ", tc_time_end - tc_time_start);

    double global_time_end = MPI_Wtime();
    world.cout0("Overall time (Triangle counting + TC): ", global_time_end - global_time_start);
  }
  return 0;
}

/*
  Questions:
  1. How to confirm correctness?
  2. How to ensure the completion of nested async?
  3. How to get the original source back? To get the triangle centrality, I need the original source vertex.
    Add the global source ID into dodgr? Is it going to affect performance or functionality?
  4. how to create a small dataset for debugging purposes?
  5. calling async_barrier() in for_all()?
  6. dodgr_edge_1d stores edges, which means there could be multiple edges for one vertex. but triangle centrality requires
    there to be only one vertex.
*/
