#include <iostream>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include "utils/cppheader.h"

using namespace std;
using namespace tbb;

const int SIZE = 1000000000;
const int GRAIN = 100000;

class ParallelMin {
private:
	int *myArray;
	
public:
	int min;
	
	ParallelMin(int *array) : myArray(array), min(INT_MAX) {}
	
	ParallelMin(ParallelMin &x, split) 
		: myArray(x.myArray), min(INT_MAX) {}
	
	void operator() (const blocked_range<int> &r) {
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
	Timer t;
	double ms;
	int result;
	int *a = new int[SIZE];
	
	fill_array(a, SIZE);
	display_array("a", a);
	
	cout << "Starting...\n";
	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();
		ParallelMin obj(a);
		parallel_reduce( blocked_range<int>(0, SIZE, GRAIN),
							obj);
		result = obj.min;
		ms += t.stop();
	}
	
	cout << "min = " << result << endl;
	cout << "avg time = " << (ms/N) << " ms\n";
	
	delete [] a;
	
	return 0;
}
