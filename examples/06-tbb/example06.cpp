// =================================================================
//
// File: example06.cpp
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval 
//				using the TBB technology. To compile:
//				g++ example06.cpp -o app -I/usr/local/lib/tbb/include -L/usr/local/lib/tbb/lib/intel64/gcc4.4 -ltbb
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;
using namespace tbb;

#define PI 3.14159265
#define RECTS 100000000 //1e8

double square(double x) {
	return x * x;
}

class Integration {
private:
	double x, dx;
	double (*fn) (double);
	double result;

public:
	Integration(double xx, double dxx, double (*f) (double)) 
		: x(xx), dx(dxx), fn(f), result(0) {}

	Integration(Integration &other, split)
		: x(other.x), dx(other.dx), fn(other.fn), result(0) {}

	double getResult() const {
		return result;
	}

	void operator() (const blocked_range<int> &r) {
		for (int i = r.begin(); i != r.end(); i++) {
			result += fn(x + (i * dx));
		}
	}

	void join(const Integration &other) {
		result += other.result;
	}
};

int main(int argc, char* argv[]) {
	double result, x, dx;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	x = 0;
	dx = PI / RECTS;

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		Integration obj(0, dx, sin);
		parallel_reduce(blocked_range<int>(0, RECTS), obj);
		result = obj.getResult() * dx;

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	cout << "result = " << fixed << setprecision(20)  << result << "\n";
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	return 0;
}
