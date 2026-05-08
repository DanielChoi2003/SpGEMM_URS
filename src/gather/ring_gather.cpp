#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/array.hpp>
#include <ygm/container/set.hpp>
#include <iostream>
#include <vector>


using std::cout, std::cin, std::endl, std::vector;


int main(int argc, char** argv){

    ygm::comm world(&argc, &argv);
    static ygm::comm &s_world = world;

    world.welcome();
    int node_size = world.layout().local_size();
    int local_id = world.layout().local_id();
    int node_id = world.layout().node_id();

    static int recv_count = 0;
    static vector<uint64_t> incoming;

    auto recv_vec = [](vector<uint64_t> vec){
        incoming = vec;
        ++recv_count;
        s_world.cout("incremented recv_count: ", recv_count);
    };

    vector<uint64_t> accum_vec = {static_cast<uint64_t>(world.rank())}; // at the end, this vector should be same across all master ranks
    incoming = accum_vec;

    if(local_id == 0){
        int right_nei = (world.rank() + node_size) % world.size();
        // run for P - 1 times
        for(int step = 0; step < node_size - 1; step++ ){
            // send its recently received vector to its right neighbor
            s_world.cout("Sending to right neighbor: ", right_nei);
            world.async(right_nei, recv_vec, incoming);
            int target = step + 1;
            // wait until it has received a vector from its left neighbor
            s_world.cout("entering loop with step ", step);
            //world.async_barrier();
            world.local_wait_until([&target]{
                //cout << "waiting to receive from left neighbor..." << endl;
                return recv_count >= target; 
            });
            accum_vec.insert(accum_vec.end(), incoming.begin(), incoming.end());
        }
    }
    world.barrier();

    if(local_id == 0){
        cout << world.rank() << "'s vector: { ";
        for(int i = 0; i < accum_vec.size(); i++){
            if(i == accum_vec.size() - 1){
                cout << accum_vec[i] << " }" << endl;
                break;
            }
            cout << accum_vec[i] << ", ";
        }
    }


    return 0;
}