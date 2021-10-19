// =================================================================
//
// File: example6.cpp
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector using Intel's TBB. To compile:
//				g++ example6.cpp -ltbb
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <climits>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "utils.h"

const int RENS = 30000; //1e5
const int COLS = 30000;

using namespace std;
using namespace tbb;

class MatrixVector {
private:
	int *M, *B, *C;

public:
	MatrixVector(int *m, int *b, int *c) : M(m), B(b), C(c) {}

	void operator() (const blocked_range<int> &r) const {
		int acum;

		acum = 0;
		for (int i = r.begin(); i != r.end(); i++) {
			acum = 0;
			for (int j = 0; j < COLS; j++) {
				acum += (M[(i * COLS) + j] * B[i]);
			}
			C[i] = acum;
		}
	}
};

int main(int argc, char* argv[]) {
	int i, j, *m, *b, *c;
	double ms;

	m = new int [RENS * COLS];
	b = new int [RENS];
	c = new int [RENS];

	for (i = 0; i < RENS; i++) {
		for (j = 0; j < COLS; j++) {
			m[(i * COLS) + j] = (j + 1);
		}
		b[i] = 1;
	}

	cout << "Starting..." << endl;
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		parallel_for(blocked_range<int>(0, RENS),  MatrixVector(m, b, c));
		/*
		MatrixVector obj(m, b, c);
		parallel_for(blocked_range<int>(0, RENS),  obj);
		*/

		ms += stop_timer();
	}
	display_array("c:", c);
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] m;
	delete [] b;
	delete [] c;
	return 0;
}
