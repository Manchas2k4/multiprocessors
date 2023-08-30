// =================================================================
//
// File: example04.cpp
// Author: Pedro Perez
// Description: This file implements the algorithm to find the 
//				minimum value in an array using the TBB technology. 
//				To compile:
//				g++ example04.cpp -o app -I/usr/local/lib/tbb/include -L/usr/local/lib/tbb/lib/intel64/gcc4.4 -ltbb
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <climits>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;
using namespace tbb;

#define SIZE 1000000000 // 1e9

class MinValue {
private:
	int *array;
	int result;

public:
	MinValue(int *a) : array(a), result(0) {}

	MinValue(MinValue &other, split): array(other.array), result(0) {}

	int getResult() const {
		return result;
	}

	void operator() (const blocked_range<int> &r) {
		result = INT_MAX;
		for (int i = r.begin(); i != r.end(); i++) {
			if (array[i] < result) {
				result = array[i];
			}
		}
	}

	void join(const MinValue &other) {
		if (other.result < result) {
			result = other.result;
		}
	}
};

int main(int argc, char* argv[]) {
	int *array, result;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	array = new int [SIZE];
	
	random_array(array, SIZE);
	display_array("array:", array);

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		MinValue obj(array);
		parallel_reduce(blocked_range<int>(0, SIZE), obj);
		result = obj.getResult();

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	cout << "result = " << result << "\n";
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	delete [] array;

	return 0;
}
