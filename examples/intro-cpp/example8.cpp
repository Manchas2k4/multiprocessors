// =================================================================
//
// File: example5.cpp
// Author: Pedro Perez
// Description: This file contains the code that implements the
//				enumeration sort algorithm. The time this implementation
//				takes ill be used as the basis to calculate the
//				improvement obtained with parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include "utils.h"

const int SIZE = 100000; //1e5

using namespace std;

// implement your class here

int main(int argc, char* argv[]) {
	int *a;
	double ms;

	a = new int[SIZE];
	random_array(a, SIZE);
	display_array("before", a);

	cout << "Starting..." << endl;
	ms = 0;
	// create object here
	for (int i = 0; i < N; i++) {
		start_timer();

		// call your method here.

		ms += stop_timer();
	}

	display_array("after", a);
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] a;
	return 0;
}
