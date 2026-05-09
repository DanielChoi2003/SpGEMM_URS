#include "ring_gather.h"


using std::cout, std::cin, std::endl, std::vector;


int main(int argc, char** argv){

    ygm::comm world(&argc, &argv);
    static ygm::comm &s_world = world;

    world.welcome();
    
    RingGather<uint64_t> RG(world);

    uint64_t my_rank = (uint64_t)world.rank();
    vector<uint64_t> local_vec = {my_rank, my_rank + 1, my_rank + 2};
  

    vector<uint64_t> global_vec = RG.Ring_Gather_Master_Rank(local_vec);

    if(world.rank0()){
        for(int i = 0; i < global_vec.size(); i++){
            cout << global_vec[i] << " ";
        }
        cout << endl;
    }
    return 0;
}