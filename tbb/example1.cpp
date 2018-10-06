#include <iostream>
#include <cstdlib>
#include <ctime>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include "utils/cppheader.h"

using namespace std; 
using namespace tbb;

const int SIZE = 1000000000;
const int GRAIN = 100000;

class ArrayInitializer {
private:
	int *myArray;
	
public:
	ArrayInitializer(int *array) : myArray(array) {
		srand(time(0));
	}
	
	void operator() (const blocked_range<int> &r) const {
		for (int i = r.begin(); i != r.end(); i++) {
			myArray[i] = (i % MAX_VALUE) + 1;
		}
	}
};

class ParallelSum {
private:
	int *myArray;
	
public:
	long sum;
	
	ParallelSum(int *array) : myArray(array), sum(0) {}
	
	ParallelSum(ParallelSum &x, split) 
		: myArray(x.myArray), sum(0) {}
	
	void operator() (const blocked_range<int> &r) {
		for (int i = r.begin(); i != r.end(); i++) {
			sum += myArray[i];
		}
	}
	
	void join(const ParallelSum &x) {
		sum += x.sum;
	}
};

int main(int argc, char* argv[]) {
	Timer t;
	double ms;
	long result;
	int *a = new int[SIZE];
	
	parallel_for( blocked_range<int>(0, SIZE, GRAIN),
	              ArrayInitializer(a) );
	display_array("a", a);
	
	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();
		ParallelSum obj(a);
		parallel_reduce( blocked_range<int>(0, SIZE, GRAIN),
	                 obj );
	    result = obj.sum;
		ms += t.stop();
	}
	cout << "sum = " << result << endl;
	cout << "avg time = " << (ms/N) << " ms" << endl;
	
	delete [] a;
	
	return 0;
}
