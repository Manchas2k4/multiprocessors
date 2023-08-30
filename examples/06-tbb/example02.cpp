// =================================================================
//
// File: example02.cpp
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//				X within the array and replaces it with Y using the 
//				TBB technology. To compile:
//				g++ example02.cpp -o app -I/usr/local/lib/tbb/include -L/usr/local/lib/tbb/lib/intel64/gcc4.4 -ltbb
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;
using namespace tbb;

#define SIZE 100000000 //1e8

class Replace {
private:
	int *array;
	int oldElement, newElement;

public:
	Replace(int *a, int x, int y) 
		: array(a), oldElement(x), newElement(y) {}

	void operator() (const blocked_range<int> &r) const {
		for (int i = r.begin(); i != r.end(); i++) {
			if (array[i] == oldElement) {
				array[i] = newElement;
			}
		}
	}
};

int main(int argc, char* argv[]) {
	int *array, *aux;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	array = new int[SIZE];
	for (int i = 0; i < SIZE; i++) {
		array[i] = 1;
	}
	display_array("before", array);
	
	aux = new int[SIZE];

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		memcpy(aux, array, sizeof(int) * SIZE);
		
		start = high_resolution_clock::now();

		parallel_for(blocked_range<int>(0, SIZE),  Replace(aux, 1, -1));

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	
	display_array("after", aux);
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	delete [] array;
	
	return 0;
}