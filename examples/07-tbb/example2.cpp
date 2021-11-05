// =================================================================
//
// File: example2.cpp
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array using Intel's TBB.
//				To compile: g++ example2.cpp -ltbb
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
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include "utils.h"

const int SIZE = 1000000000; //1e9

using namespace std;
using namespace tbb;

class MinValue {
private:
	int *array, result;

public:
	MinValue(int *a) : array(a), result(INT_MAX) {}

	MinValue(MinValue &x, split) : array(x.array), result(INT_MAX) {}

	int getResult() const {
		return result;
	}

	void operator() (const blocked_range<int> &r) {
		for (int i = r.begin(); i != r.end(); i++) {
			result = min(result, array[i]);
		}
	}

	void join(const MinValue &x) {
		result = min(result, x.result);
	}
};

int main(int argc, char* argv[]) {
	int *a, pos, result;
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
	for (int i = 0; i < N; i++) {
		start_timer();

		MinValue obj(a);
		parallel_reduce(blocked_range<int>(0, SIZE), obj);
		result = obj.getResult();

		ms += stop_timer();
	}
	cout << "result = " << result << endl;
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] a;
	return 0;
}
