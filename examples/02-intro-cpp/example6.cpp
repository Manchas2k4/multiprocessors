// =================================================================
//
// File: example6.cpp
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector. The time this implementation takes will
//				be used as the basis to calculate the improvement
//				obtained with parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <climits>
#include "utils.h"

const int RENS = 30000; //1e5
const int COLS = 30000;

using namespace std;

class MatrixVector {
private:
	int *M, *B, *C;

public:
	MatrixVector(int *m, int *b, int *c) : M(m), B(b), C(c) {}

	void calculate() {
		int i, j, acum;

		for (i = 0; i < RENS; i++) {
			acum = 0;
			for (j = 0; j < COLS; j++) {
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
	MatrixVector obj(m, b, c);
	for (i = 0; i < N; i++) {
		start_timer();

		obj.calculate();

		ms += stop_timer();
	}
	display_array("c:", c);
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] m;
	delete [] b;
	delete [] c;
	return 0;
}
