#include "shm_counting_set.h"

int main(int argc, char **argv){

    ygm::comm world(&argc, &argv);
    static ygm::comm &s_world = world;

    ygm::container::map<std::pair<int, int>, int> accumulator(world);

    shm_counting_set cache(world, accumulator);

    int garbo;
    cache.cache_insert({0, 0}, world.rank(), garbo);

    world.barrier();

    accumulator.for_all([](std::pair<int, int> index, int value){
        s_world.cout("index: ", index.first, ", ", index.second, ". ", "value: ", value);
    });
}