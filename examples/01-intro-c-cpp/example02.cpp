// =================================================================
//
// File: example01.c 
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//				X within the array and replaces it with Y. The time 
//				it takes to implement this will be used as the basis 
//				for calculating the improvement obtained with parallel 
//				technologies. The time this implementation takes.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 100000000 //1e8

void replace(int *array, int size, int x, int y) {
	int i;

	for (i = 0; i < size; i++) {
		if (array[i] == x) {
			array[i] = y;
		}
	}
}

int main(int argc, char* argv[]) {
	int *array;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	array = new int[SIZE];
	fill_array(array, SIZE);
	display_array("before", array);

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		replace(array, SIZE, 1, -1);

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	
	display_array("after", array);
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	delete [] array;
	
	return 0;
}