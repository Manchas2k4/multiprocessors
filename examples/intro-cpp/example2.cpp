// =================================================================
//
// File: example2.cpp
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array. The time this
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
#include <climits>
#include <algorithm>
#include "utils.h"

const int SIZE = 1000000000; //1e9

using namespace std;

class MinValue {
private:
	int *array, size, result;

public:
	MinValue(int *a, int s) : array(a), size(s) {}

	int getResult() const {
		return result;
	}

	void calculate() {
		result = INT_MAX;
		for (int i = 0; i < size; i++) {
			result = min(result, array[i]);
		}
	}
};

int main(int argc, char* argv[]) {
	int *a, pos;
	double ms;

	a = new int[SIZE];
	random_array(a, SIZE);
	display_array("a", a);

	srand(time(0));
	pos = rand() % SIZE;
	printf("Setting value 0 at %i\n", pos);
	a[pos] = 0;

	cout << "Starting..." << endl;
	ms = 0;
	MinValue obj(a, SIZE);
	for (int i = 0; i < N; i++) {
		start_timer();

		obj.calculate();

		ms += stop_timer();
	}
	cout << "result = " << obj.getResult() << endl;
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] a;
	return 0;
}
