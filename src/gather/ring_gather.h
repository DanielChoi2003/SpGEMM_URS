#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/array.hpp>
#include <ygm/container/set.hpp>
#include <iostream>
#include <vector>
#include <queue>

using std::cout, std::cin, std::endl, std::vector, std::queue;


template <typename Value>
class RingGather{
    static_assert(std::is_trivially_copyable_v<Value>);
public:
    RingGather(ygm::comm& c) : 
                m_comm(c), 
                m_local_size(m_comm.layout().local_size()),
                m_local_id(m_comm.layout().local_id()),
                m_node_id(m_comm.layout().node_id()),
                m_node_size(m_comm.layout().node_size())
                
    {
        pthis.check(m_comm);
    }


    vector<Value> Ring_Gather_Master_Rank(vector<Value> local_vec){
        m_comm.welcome();


        // incoming should be a queue, not a vector, because it can be overwritten by a second call

        auto recv_vec = [](auto self, vector<uint64_t> vec){
            self->incoming.push(vec);
            self->recv_count++;
        };

        vector<Value> recv = local_vec;

        m_comm.barrier();

        if(m_local_id == 0){
            int right_nei = (m_comm.rank() + m_local_size) % m_comm.size();
            // run for P - 1 times
            for(int step = 0; step < m_node_size - 1; step++ ){
                // send its recently received vector to its right neighbor
                m_comm.async(right_nei, recv_vec, pthis, recv);
                int target = step + 1;
                // wait until it has received a vector from its left neighbor
                m_comm.local_wait_until([this, &target]{
                    return recv_count >= target; 
                });
                cout << incoming.size() << endl;
                recv = incoming.front();
                incoming.pop();
                cout << "popped" << endl;
                accum.insert(accum.end(), recv.begin(), recv.end());
            }
        }
        m_comm.barrier();

        return accum;
    }




private:

    ygm::comm& m_comm;

    size_t m_local_size = -1;
    size_t m_local_id = -1;
    size_t m_node_id = -1;
    size_t m_node_size = -1;

    // should not be function-level static variables, 
    // else, these would be visible from other simultaneous ring gathers, which is 
    // not what we want!
    vector<Value> accum;
    // incoming should be a queue, not a vector, because it can be overwritten by a second call
    queue<vector<Value>> incoming;
    int recv_count = 0;

    typename ygm::ygm_ptr<RingGather>                     pthis;

};