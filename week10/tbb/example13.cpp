#include <iostream>
#include <tbb/task_scheduler_init.h>
#include <tbb/task_group.h>

using namespace std;
using namespace tbb;

class TaskOne {
private:
	int id;
	
public:
	TaskOne(int i) : id(i) {}
	
	void operator() () const {
		cout << "TaskOne id = " << id << "\n";
	}
};

class TaskTwo {
private:
	int id;
	int max;
	
public:
	TaskTwo(int i, int n) : id(i), max(n) {}
	
	void operator() () const {
		for (int i = 0; i < max; i++) {
			cout << "TaskTwo id = " << id << " i = " << i << "\n";
		}
	}
};

int main(int argc, char* argv[]) {
	task_group tg;
	
	tg.run(TaskOne(1));
	tg.run(TaskTwo(2, 10));
	tg.run(TaskOne(3));
	tg.run(TaskTwo(4, 30));
	tg.wait();
	
	return 0;
};
