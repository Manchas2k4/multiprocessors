// =================================================================
//
// File: example01.cpp
// Author: Pedro Perez
// Description: This file implements the addition of two vectors 
//				using the TBB technology. To compile:
//				g++ example01.cpp -o app -I/usr/local/lib/tbb/include -L/usr/local/lib/tbb/lib/intel64/gcc4.4 -ltbb
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;
using namespace tbb;

#define SIZE 10000000 // 1e7

class AddVectors {
private:
	int *vectorR, *vectorA, * vectorB;

public:
	AddVectors(int *result, int *a, int *b) 
		: vectorR(result), vectorA(a), vectorB(b) {} 

	void operator() (const blocked_range<int> &r) const {
		for (int i = r.begin(); i != r.end(); i++) {
			vectorR[i] = vectorA[i] + vectorB[i];
		}
	}
};

int main(int argc, char* argv[]) {
	int *a, *b, *c;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	a = new int [SIZE];
	b = new int [SIZE];
	c = new int [SIZE];

	fill_array(a, SIZE);
	display_array("a:", a);
	fill_array(b, SIZE);
	display_array("b:", b);

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		parallel_for(blocked_range<int>(0, SIZE),  AddVectors(c, a, b));

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	display_array("c:", c);
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	delete [] a;
	delete [] b;
	delete [] c;

	return 0;
}
