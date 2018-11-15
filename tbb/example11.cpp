#include <iostream>
#include <tbb/task_scheduler_init.h>
#include <tbb/task_group.h>
#include <tbb/concurrent_queue.h>

using namespace std;
using namespace tbb;

class Productor {
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
			cout << "Productor id = " << id 
			     << " value = " << val << "\n";
		}
	}
};

class Consumidor {
private:
	int id;
	concurrent_bounded_queue<int> &myQueue;
	
public:
	Consumidor(int i, concurrent_bounded_queue<int> &queue) 
		: id(i), myQueue(queue) {}
		
	void operator() () const {
		int val;
		
		for (int i = 0; i < 10; i++) {
			cout << "Consumidor id = " << id 
			     << "tomando del queue...\n";
			myQueue.pop(val);
			cout << "Consumidor id = " << id 
				 << " value = " << val << "\n";
		}
	}
}; 

int main(int argc, char* argv[]) {
	task_group tg;
	concurrent_bounded_queue<int> queue;
	
	tg.run(Consumidor(1, queue));
	tg.run(Consumidor(2, queue));
	sleep(10);
	tg.run(Productor(1, queue));
	tg.run(Productor(2, queue));
	tg.wait();
	return 0;
}
