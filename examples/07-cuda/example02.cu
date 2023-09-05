// =================================================================
//
// File: example02.cpp
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//				X within the array and replaces it with Y using CUDA 
//				technology.
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 100000000 //1e8
#define THREADS 512
#define BLOCKS	max(32, ((SIZE / THREADS) + 1))

__global__ void my_replace(int *source, int *destination, int *oldElement, int *newElement) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i < SIZE) {
		if (source[i] != *oldElement) {
			destination[i] = source[i];
		} else {
			destination[i] = *newElement;
		}
	}
}

int main(int argc, char* argv[]) {
	int *array, *aux, oldElement, newElement;
	int *d_array, *d_aux, *d_oldElement, *d_newElement;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	array = new int [SIZE];
	for (int i = 0; i < SIZE; i++) {
		array[i] = 1;
	}
	display_array("before", array);

	aux = new int [SIZE];
	oldElement = 1;
	newElement = -1;

	cudaMalloc((void**) &d_array, SIZE * sizeof(int));
	cudaMalloc((void**) &d_aux, SIZE * sizeof(int));
	cudaMalloc((void**) &d_oldElement, sizeof(int));
	cudaMalloc((void**) &d_newElement, sizeof(int));

	cudaMemcpy(d_array, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_aux, aux, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_oldElement, &oldElement, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_newElement, &newElement, sizeof(int), cudaMemcpyHostToDevice);

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		my_replace<<<BLOCKS, THREADS>>>(d_array, d_aux, d_oldElement, d_newElement);

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}

	cudaMemcpy(aux, d_aux, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("after:", aux);
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	cudaFree(d_array);
	cudaFree(d_aux);
	cudaFree(d_oldElement);
	cudaFree(d_newElement);

	delete [] array;
	delete [] aux;

	return 0;
}
