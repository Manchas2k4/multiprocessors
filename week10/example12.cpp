#include <iostream>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_do.h>
#include <list>
#include "utils/cppheader.h"

using namespace std;
using namespace tbb;

const int SIZE = 100000000;
const int GRAING = 100000;

class Apply {
public:
	void operator() (int &item) const {
		item *= 2;
	}
};

void one_thread(list<int> &lst) {
	list<int>::iterator itr;
	
	for (itr = lst.begin(); itr != lst.end(); itr++) {
		(*itr) *= 2;
	}
}

int main(int argc, char* argv[]) {
	Timer t;
	list<int> lst;
	double ms;
	
	for (int i = 0; i < SIZE; i++) {
		lst.push_back(i);
	}
	
	t.start();
	one_thread(lst);
	ms = t.stop();
	cout << "one thread time = " << ms << " ms\n";
	
	t.start();
	parallel_do(lst.begin(), lst.end(), Apply());
	ms = t.stop();
	cout << "tbb time = " << ms << " ms\n";
	
	return 0;
}
