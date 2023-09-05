// =================================================================
//
// File: example01c.cpp
// Author: Pedro Perez
// Description: This file implements the addition of two vectors
//				using CUDA technology. In this case, we use 
//				a ''matrix'' of cores.
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

#define SIZE 10000000 // 1e7
#define THREADS 512
#define BLOCKS	max(32, ((SIZE / THREADS) + 1))

__global__ void add(int *a, int *b, int *c) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i < SIZE) {
		c[i] = a[i] + b[i];
	}
}

int main(int argc, char* argv[]) {
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	a = new int [SIZE];
	b = new int [SIZE];
	c = new int [SIZE];

	fill_array(a, SIZE);
	display_array("a:", a);
	fill_array(b, SIZE);
	display_array("b:", b);

	cudaMalloc((void**) &d_a, SIZE * sizeof(int));
	cudaMalloc((void**) &d_b, SIZE * sizeof(int));
	cudaMalloc((void**) &d_c, SIZE * sizeof(int));

	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		add<<<BLOCKS, THREADS>>>(d_a, d_b, d_c);

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}

	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("c:", c);
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);

	delete [] a;
	delete [] b;
	delete [] c;

	return 0;
}
