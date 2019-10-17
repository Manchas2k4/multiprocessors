/* This code calculates the sum of all elements in an array */
#include <iostream>
#include <cmath>
#include "utils/cppheader.h"
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

const int SIZE = 1000000000; //1e9
const int GRAIN = 100000; //1e5

using namespace std;
using namespace tbb;

class SumArray {
private:
	int *myArray;
	
public:
	long result;

	SumArray(int *array) : myArray(array), result(0) {}

	SumArray(SumArray &x, split) : myArray(x.myArray), result(0) {}
	
	void operator() (const blocked_range<int> &r) {
		result = 0;
		for (int i = r.begin(); i != r.end(); i++) {
			result += myArray[i];
		}
	}

	void join(const SumArray &x) {
		result += x.result;
	}
};

int main(int argc, char* argv[]) {
	long result = 0;
	double ms;
	Timer t;
	int *a;
	
	a = new int[SIZE];
	fill_array(a, SIZE);
	display_array("a", a);
	
	
	ms = 0;
	cout << "Starting..." << endl;
	for (int i = 0; i < N; i++) {
		t.start();

		SumArray obj(a);
		parallel_reduce(blocked_range<int>(0, SIZE, GRAIN), obj);
		result = obj.result;

		ms += t.stop();
	}
	cout << "sum = " << result << endl;
	cout << "avg time = " << (ms / N) << " ms" << endl;
	
	delete [] a;
	return 0;
}
