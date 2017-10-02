#include <iostream>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "utils/cppheader.h"

using namespace std;
using namespace tbb;

const int SIZE = 100000000;
const int GRAIN = 100000;

class AddTwoArrays {
private:
	int *myA, *myB, *myC;
	
public:
	AddTwoArrays(int *c, int *a, int *b)
		: myC(c), myA(a), myB(b) {}
		
	void operator() (const blocked_range<int> &r) const {
		for (int i = r.begin(); i != r.end(); i++) {
			myC[i] = myA[i] + myB[i];
		}
	}
}; 

int main(int argc, char* argv[]) {
	Timer t;
	double ms;
	
	int *a = new int[SIZE];
	fill_array(a, SIZE);
	display_array("a", a);
	
	int *b = new int[SIZE];
	fill_array(b, SIZE);
	display_array("b", b);
	
	int *c = new int[SIZE];
	
	cout << "Starting...\n";
	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();
		parallel_for( blocked_range<int>(0, SIZE, GRAIN),
						AddTwoArrays(c, a, b) );
		ms += t.stop();
	}
	display_array("c", c);
	cout << "avg time = " << (ms/N) << " ms\n";
	
	delete [] a;
	delete [] b;
	delete [] c;
	
	return 0;
}
