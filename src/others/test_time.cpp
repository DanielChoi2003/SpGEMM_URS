#include <ygm/comm.hpp>
#include <ygm/container/array.hpp>
#include <iostream>
#include <vector>

using std::vector, std::cout, std::cin, std::endl;

int main(int argc, char **argv){

    ygm::comm world(&argc, &argv);
    static ygm::comm &s_world = world;

    

    ygm::container::array<int> arr(world);
    

    return 0;
}