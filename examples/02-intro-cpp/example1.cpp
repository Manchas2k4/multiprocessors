// =================================================================
//
// File: example1.cpp
// Author: Pedro Perez
// Description: This file contains the code that adds all the
//				elements of an integer array. The time this
//				implementation takes will be used as the basis to
//				calculate the improvement obtained with parallel
//				technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include "utils.h"

const int SIZE = 1000000000; //1e9

using namespace std;

class AddArray {
private:
	int *array, size;
	double result;

public:
	AddArray(int *a, int s) : array(a), size(s) {}

	double getResult() const {
		return result;
	}

	void calculate () {
		result = 0;
		for (int i = 0; i < size; i++) {
			result += array[i];
		}
	}
};

int main(int argc, char* argv[]) {
	int *a;
	double ms, result;
	// #include <cstdlib>
	// #include <cstdio>
	a = new int[SIZE];
	fill_array(a, SIZE);
	display_array("a", a);

	cout << "Starting..." << endl;
	ms = 0;
	AddArray obj(a, SIZE);
	for (int i = 0; i < N; i++) {
		start_timer();

		obj.calculate();

		ms += stop_timer();
	}
	cout << "sum = " << (long int) obj.getResult() << endl;
	//printf("%.15lf")
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] a;
	return 0;
}
