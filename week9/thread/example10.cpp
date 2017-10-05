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

class ParallelOcurrences {
private:
	int *myArray;
	int myValue;
	
public:
	int total;
	
	ParallelOcurrences(int *array, int value) 
		: myArray(array), myValue(value), total(0) {}
	
	ParallelOcurrences(ParallelOcurrences &x, split) 
		: myArray(x.myArray), myValue(x.myValue), total(0) {}
	
	void operator() (const blocked_range<int> &r) {
		for (int i = r.begin(); i != r.end(); i++) {
			if (myValue == myArray[i]) {
				total++;
			}
		}
	}
	
	void join(const ParallelOcurrences &x) {
		total += x.total;
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
		ParallelOcurrences obj(a, 100);
		parallel_reduce( blocked_range<int>(0, SIZE, GRAIN),
							obj);
		result = obj.total;
		ms += t.stop();
	}
	
	cout << "ocurrences = " << result << endl;
	cout << "avg time = " << (ms/N) << " ms\n";
	
	delete [] a;
	
	return 0;
}
