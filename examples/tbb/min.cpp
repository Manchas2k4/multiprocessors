/* This code calculates the sum of all elements in an array */
#include <iostream>
#include "utils/cppheader.h"
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

const int SIZE = 1000000000; //1e9
const int GRAIN = 100000; //1e5

using namespace std;
using namespace tbb;

class ParallelMin {
private:
	int *myArray;
	
public:
	int min;

	ParallelMin(int *array) : myArray(array), min(INT_MAX) {}

	ParallelMin(ParallelMin &x, split) : myArray(x.myArray), min(INT_MAX) {}
	
	void operator() (const blocked_range<int> &r) {
		min = INT_MAX;
		for (int i = r.begin(); i != r.end(); i++) {
			if (min > myArray[i]) {
				min = myArray[i];
			}
		}
	}

	void join(const ParallelMin &x) {
		if (min > x.min) {
			min = x.min;
		}
	}
};

int main(int argc, char* argv[]) {
	double ms;
	Timer t;
	int *a, min;
	
	a = new int[SIZE];
	fill_array(a, SIZE);
	display_array("a", a);
	
	
	ms = 0;
	cout << "Starting..." << endl;
	for (int i = 0; i < N; i++) {
		t.start();

		ParallelMin obj(a);
		parallel_reduce(blocked_range<int>(0, SIZE, GRAIN), obj);
		min = obj.min;

		ms += t.stop();
	}
	cout << "min = " << min << endl;
	cout << "avg time = " << (ms / N) << " ms" << endl;
	
	delete [] a;
	return 0;
}
