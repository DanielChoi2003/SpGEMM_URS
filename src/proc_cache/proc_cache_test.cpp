#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/array.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/io/csv_parser.hpp>
#include <iostream>
#include "proc_cache.hpp"



int main(int argc, char **argv){

    ygm::comm world(&argc, &argv);
    static ygm::comm &s_world = world;

    

    return 0;
}