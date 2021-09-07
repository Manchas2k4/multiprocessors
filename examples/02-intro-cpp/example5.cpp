// =================================================================
//
// File: example5.cpp
// Author: Pedro Perez
// Description: This file contains the code that implements the
//				bubble sort algorithm. The time this implementation takes
//				will be used as the basis to calculate the improvement
//				obtained with parallel technologies.
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

const int SIZE = 100000; //1e4

using namespace std;

class BubbleSort {
private:
	int *A, size;

	void swap(int *a, int i, int j) {
		int aux = a[i];
		a[i] = a[j];
		a[j] = aux;
	}

public:
	BubbleSort(int *a, int s) : A(a), size(s) {}

	void doTask() {
		for(int i = size - 1; i > 0; i--){
			for(int j = 0; j < i; j++){
				if(A[j] > A[j + 1]){
					swap(A, j, j + 1);
				}
			}
		}
	}
};

int main(int argc, char* argv[]) {
	int *a, *aux;
	double ms;

	a = new int[SIZE];
	random_array(a, SIZE);
	display_array("before", a);

	aux = new int[SIZE];

	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		start_timer();

		memcpy(aux, a, sizeof(int) * SIZE);
		BubbleSort obj(aux, SIZE);
		obj.doTask();

		ms += stop_timer();
	}

	display_array("after", aux);
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] a;
	delete [] aux;
	return 0;
}
