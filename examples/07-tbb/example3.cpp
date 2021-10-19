// =================================================================
//
// File: Example3.cpp
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval
//				using Intel's TBB. To compile:
//				g++ example3.cpp -ltbb
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include "utils.h"

const double PI = 3.14159265;
const int RECTS = 1000000000; //1e9

using namespace std;
using namespace tbb;

double function(double x) {
	return sin(x);
}

class Integration {
private:
	double x, dx, result;
	double (*func) (double);

public:
	Integration(double xx, double dxx, double (*fn) (double))
		: x(xx), dx(dxx), func(fn), result(0) {}

	Integration(Integration &obj, split)
		: x(obj.x), dx(obj.dx), func(obj.func), result(0) {}

	double getResult() const {
		return result * dx;
	}

	// 	void operator() (const blocked_range<int> &r) const {
	void operator() (const blocked_range<int> &r) {
		for (int i = r.begin(); i != r.end(); i++) {
			result += func(x + (i * dx));
		}
	}

	void join(const Integration &x) {
		result += x.result;
	}
};

int main(int argc, char* argv[]) {
	double ms;
	double x, dx, result;

	x = 0;
	dx = (PI - 0.0) / RECTS;

	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		start_timer();

		Integration obj(x, dx, function);
		parallel_reduce(blocked_range<int>(0, RECTS), obj);
		result = obj.getResult();

		ms += stop_timer();
	}
	cout << "result = " << setprecision(15) << result << endl;
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	return 0;
}
