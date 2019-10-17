#include <iostream>
#include <tbb/task_scheduler_init.h>
#include <tbb/task_group.h>

using namespace std;
using namespace tbb;

class MyTask {
private:
	int id;

public:
	MyTask(int i) : id(i) {}

	void operator() () const {
		cout << "Thread id = " << id << "\n";
	}
};

int main(int argc, char* argv[]) {
	task_group tg;

	for (int i = 0; i < 100; i++) {
		tg.run(MyTask(i + 1));
	}
	tg.wait();
	return 0;
}
