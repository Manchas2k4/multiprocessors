// =================================================================
//
// File: example13.cpp
// Author: Pedro Perez
// Description: This file implements the quick sort algorithm. The
//				time this implementation takes will be used as the
//				basis to calculate the improvement obtained with
//				parallel technologies. To compile:
//				g++ example13.cpp -ltbb
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include <tbb/parallel_invoke.h>
#include "utils.h"

const int SIZE = 100000000; //1e8

using namespace std;
using namespace tbb;

class QuickSort {
private:
	int *A, size;

	void swap(int *a, int i, int j) {
		int aux = a[i];
		a[i] = a[j];
		a[j] = aux;
	}

	int findPivot(int low, int high) {
		for (int i = low + 1; i <= high; i++) {
			if (A[low] > A[i]) {
				return A[low];
			} else if (A[low] < A[i]){
				return A[i];
			}
		}
		return -1;
	}

	int makePartition(int low, int high, int pivot) {
		int i, j;

		i = low;
		j = high;
		while (i < j) {
			swap(A, i , j);
			while (A[i] < pivot) {
				i++;
			}
			while (A[j] >= pivot) {
				j--;
			}
		}
		return i;
	}

	void quick(int low, int high) {
		int pivot, pos;

		pivot = findPivot(low, high);
		if (pivot != -1) {
			pos = makePartition(low, high, pivot);
			parallel_invoke (
				[=] { quick(low, pos - 1); },
				[=] { quick(pos, high); }
			);
		}
	}

public:
	QuickSort(int *a, int s) {
		size = s;
		A = new int[size];
		memcpy(A, a, sizeof(int) * SIZE);
	}

	~QuickSort() {
		delete [] A;
	}

	int* getSortedArray() const {
		return A;
	}

	void doTask() {
		quick(0, size - 1);
	}
};

int main(int argc, char* argv[]) {
	int *a;
	double ms;

	a = new int[SIZE];
	random_array(a, SIZE);
	display_array("before", a);

	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		start_timer();

		QuickSort obj(a, SIZE);
		obj.doTask();

		ms += stop_timer();

		if (i == (N - 1)) {
			memcpy(a, obj.getSortedArray(), sizeof(int) * SIZE);
		}
	}

	display_array("after", a);
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] a;
	return 0;
}
