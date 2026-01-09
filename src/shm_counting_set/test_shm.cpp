#include "shm_counting_set.h"

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

int main(int argc, char **argv){

    ygm::comm world(&argc, &argv);
    static ygm::comm &s_world = world;

    ygm::container::map<map_key, int> accumulator(world);

    shm_counting_set cache(world, accumulator);

    int garbo;
    auto garb_ptr = world.make_ygm_ptr(garbo);
    cache.cache_insert({0, 0}, 1, garb_ptr);
    world.barrier();
    cache.value_cache_flush_all();
    world.barrier();

    accumulator.for_all([](map_key index, int value){
        s_world.cout("index: ", index.x, ", ", index.y, ". ", "value: ", value);
    });
}