// =================================================================
//
// File: example12.cpp
// Author: Pedro Perez
// Description: This file implements the merge sort algorithm. The
//				time this implementation takes will be used as the
//				basis to calculate the improvement obtained with
//				parallel technologies.
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

const int SIZE = 100000000; //1e8
const int GRAIN = 1000; // 1e3

using namespace std;

class MergeSort {
private:
	int *A, *B, size;

	void swap(int *a, int i, int j) {
		int aux = a[i];
		a[i] = a[j];
		a[j] = aux;
	}

	void copyArray(int low, int high) {
		int length = high - low + 1;
		memcpy(A + low, B + low, sizeof(int) * length);
	}

	void merge(int low, int mid, int high) {
		int i, j, k;

		i = low;
		j = mid + 1;
		k = low;
		while(i <= mid && j <= high){
			if(A[i] < A[j]){
				B[k] = A[i];
				i++;
			}else{
				B[k] = A[j];
				j++;
			}
			k++;
		}
		for(; j <= high; j++){
			B[k++] = A[j];
		}

		for(; i<= mid; i++){
			B[k++] = A[i];
		}
	}

	void split(int low, int high) {
		int  mid, size, i, j;

		size = high - low + 1;
		if(size < GRAIN) {
			for(i = low + 1; i < size; i++){
				for(j = i; j > low && A[j] < A[j - 1]; j--){
					swap(A, j, j - 1);
				}
			}
			return;
		}

		mid = low + ((high - low) / 2);
		split(low, mid);
		split(mid +1, high);
		merge(low, mid, high);
		copyArray(low, high);
	}

public:
	MergeSort(int *a, int s) {
		size = s;
		A = new int[size];
		B = new int[size];

		memcpy(A, a, sizeof(int) * SIZE);
	}

	~MergeSort() {
		delete [] A;
		delete [] B;
	}

	int* getSortedArray() {
		return A;
	}

	void doTask() {
		split(0, size - 1);
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
		MergeSort obj(a, SIZE);
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
