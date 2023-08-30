// =================================================================
//
// File: example03.cpp
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector using the TBB technology. To compile:
//				g++ example03.cpp -o app -I/usr/local/lib/tbb/include -L/usr/local/lib/tbb/lib/intel64/gcc4.4 -ltbb
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

#define RENS 10000
#define COLS 10000

class MultVectMat {
private:
	int *matrix, *vectorB, * vectorC;

public:
	MultVectMat(int *m, int *b, int *c) 
		: matrix(m), vectorB(b), vectorC(c) {}

	void operator() (const blocked_range<int> &r) const {
		for (int i = r.begin(); i != r.end(); i++) {
			int acum = 0;
			for (int j = 0; j < COLS; j++) {
				acum += (matrix[(i * COLS) + j] * vectorB[i]);
			}
			vectorC[i] = acum;
		}
	}
};

int main(int argc, char* argv[]) {
	int *m, *b, *c;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	m = new int[RENS * COLS];
	b = new int [RENS];
	c = new int [RENS];

	for (int i = 0; i < RENS; i++) {
		for (int j = 0; j < COLS; j++) {
			m[(i * COLS) + j] = (j + 1);
		}
		b[i] = 1;
	}

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		parallel_for(blocked_range<int>(0, RENS),  MultVectMat(m, b, c));

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	display_array("c:", c);
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	delete [] m;
	delete [] b;
	delete [] c;

	return 0;
}
