#include <iostream>
#include <cstdlib>
#include <ctime>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
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

class Squares {
private:
	int *myArray;
	
public:
	Squares(int *array) : myArray(array) {
	}
	
	void operator() (const blocked_range<int> &r) const {
		for (int i = r.begin(); i != r.end(); i++) {
			myArray[i] *= myArray[i];
		}
	}
};

int main(int argc, char* argv[]) {
	Timer t;
	double ms;
	
	int *a = new int[SIZE];
	parallel_for( blocked_range<int>(0, SIZE, GRAIN),
	              ArrayInitializer(a) );
	display_array("a", a);
	
	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();
		parallel_for( blocked_range<int>(0, SIZE, GRAIN),
	              ArrayInitializer(a) );
		parallel_for( blocked_range<int>(0, SIZE, GRAIN),
	              Squares(a) );
		ms += t.stop();
	}
	display_array("a", a);
	cout << "avg time = " << (ms/N) << " ms" << endl;
	
	delete [] a;
	
	return 0;
}
