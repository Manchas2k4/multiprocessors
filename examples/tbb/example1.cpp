/* This code calculates an approximation of PI */

#include <iostream>
#include "utils/cppheader.h"
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

using namespace std;
using namespace tbb;

const long NUM_RECTS = 100000000; //1e8
const int GRAIN = 100000; //1e6

class CalculatingPi {
public:
	double sum;

public:
	CalculatingPi() : sum(0) {}

	CalculatingPi(CalculatingPi &x, split) 
		: sum(0) {} 

	void operator() (const blocked_range<int> &r) {
		double mid, height, width;
		
		sum = 0;
		width = 1.0 / (double) NUM_RECTS;
		for (int i = r.begin(); i != r.end(); i++) {
			mid = (i + 0.5) * width;
			height = 4.0 / (1.0 + (mid * mid));
			sum += height;
		}
	}

	void join(const CalculatingPi &x) {
		sum += x.sum;
	}
};

int main(int argc, char* argv[]) {
	Timer t;
	double ms, area;

	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();

		CalculatingPi cp;
		parallel_reduce(blocked_range<int>(0, NUM_RECTS, GRAIN), cp);
		area = cp.sum * (1.0 / (double) NUM_RECTS);
		ms += t.stop();
	}
	cout << "PI = " << area << endl;
	cout << "avg time = " << (ms / N) << " ms" << endl;

	return 0;
}
