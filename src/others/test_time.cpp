#include <ygm/comm.hpp>
#include <iostream>
#include <vector>

using std::vector, std::cout, std::cin, std::endl;

int main(int argc, char **argv){

    ygm::comm world(&argc, &argv);
    static ygm::comm &s_world = world;

    constexpr int iters = 100;
    int msg_size = 1024;
    //1048576
    for (int step = 0; step < 10; ++step) {
        msg_size *= 2;
        std::vector<char> msg(msg_size);

        double start = 0.0;
        double total = 0.0;

        auto start_ptr = world.make_ygm_ptr(start);
        auto total_ptr = world.make_ygm_ptr(total);
        auto pong = [](std::vector<char> payload, auto start_ptr, auto total_ptr) {
            s_world.async(0, [](auto start_ptr, auto total_ptr) {
                double end = MPI_Wtime();
                *total_ptr += (end - *start_ptr);
            }, start_ptr, total_ptr );
        };

        if (world.rank() == 0) {
            for (int i = 0; i < iters; ++i) {
                start = MPI_Wtime();
                world.async(1, pong, msg, start_ptr, total_ptr);
            }

            double avg_rtt = total / iters;
            std::cout << msg_size << " bytes: "
                      << (avg_rtt / 2.0) << " sec one-way\n";
        }

    }



    


    return 0;
}