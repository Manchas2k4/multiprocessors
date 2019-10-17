#include <iostream>
#include <tbb/task_scheduler_init.h>
#include <tbb/task_group.h>
#include <tbb/concurrent_queue.h>

using namespace std;
using namespace tbb;

class Productor
private: 
	concurrent_bounded_queue<int> &myQueue;
	int id;

public:
	Productor(int i, concurrent_bounded_queue<int> &queue)
		: id(i), myQueue(queue) {}

	void operator() () const {
		int val;
		for (int i = 0; i < 10; i++) {
			val = (id * 10) + i;
			myQueue.push(val);
			cout << "Productor id = " << id << " value = " << val << "\n";
		}
	}
};
