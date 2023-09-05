// =================================================================
//
// File: example03.cu
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector using CUDA technology.
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

#define RENS    10000
#define COLS    10000
#define THREADS 512
#define BLOCKS	min(32, (((RENS * COLS) / THREADS) + 1))

__global__ void matrix_vector(int *m, int *b, int *c) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int j, sum = 0;

    while (tid < RENS){
        sum = 0;
        for(j = 0; j < COLS; j++) {
            sum += (m[(tid * COLS) + j] * b[tid]);
        }
        c[tid] = sum;
        
        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char* argv[]) {
    int i, j, *m, *b, *c;
    int *d_m, *d_b, *d_c;
    
    // These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	m = new int[RENS* COLS];
	b = new int[RENS];
	c = (int*) malloc(sizeof(int) * RENS);

    for (i = 0; i < RENS; i++) {
        for (j = 0; j < COLS; j++) {
            m[(i * COLS) + j] = (j + 1);
        }
        b[i] = 1;
    }

    cudaMalloc((void**)&d_m, sizeof(int) * RENS* COLS);
    cudaMalloc((void**)&d_b, sizeof(int) * RENS);
    cudaMalloc((void**)&d_c, sizeof(int) * RENS);

    cudaMemcpy(d_m, m, sizeof(int) * RENS* COLS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * RENS, cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        matrix_vector<<<BLOCKS, THREADS>>>(d_m, d_b, d_c);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    cudaMemcpy(c, d_c, sizeof(int) * RENS, cudaMemcpyDeviceToHost);

    display_array("c:", c);
    cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

    cudaFree(d_m); 
    cudaFree(d_b); 
    cudaFree(d_c);
    
    
    delete [] m;
	delete [] b;
	delete [] c;

    return 0;
}