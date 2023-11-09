// =================================================================
//
// File: example08.cu
// Author: Pedro Perez
// Description: This file implements the merge sort algorithm using 
//				CUDA technology.
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
#include <climits>
#include <cuda_runtime.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

typedef unsigned int uint;

#define SIZE 10000000 //1e7

__global__ void gpu_merge(int *array, int *aux, uint blockSize, const uint last) {
	int id, start, mid, end, left, right, i;
	
	id = threadIdx.x;
	start = blockSize * id;
	mid = start + (blockSize / 2) - 1;
	end = start + blockSize - 1;
	
	left = start;
	right = mid + 1;
	i = start;
	
	if (end > last) {
		end = last;
	}
	
	if (start == end || end <= mid) {
		return;
	}
	
	while (left <= mid && right <= end) {
		if (array[left] <= array[right]) {
			aux[i++] = array[left++];
		} else {
			aux[i++] = array[right++];
		}
	}
	
	while (left <= mid) {
		aux[i++] = array[left++];
	}
	
	while (right <= end) {
		aux[i++] = array[right++];
	}
	
	/*
	for (int i = start; i <= end; i++) {
		array[i] = aux [i];
	}
	*/
}

void merge_sort(int *array, uint size) {
	int *d_array, *d_temp, *A, *B;
	uint threadCount, last;
	
	cudaMalloc( (void**) &d_array, size * sizeof(int) );
	cudaMalloc( (void**) &d_temp, size * sizeof(int) );
	
	cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_temp, array, size * sizeof(int), cudaMemcpyHostToDevice);
	
	A = d_array;
	B = d_temp;
	
	last = size - 1;
	for (uint blockSize = 2; blockSize < (2 * size); blockSize *= 2) {
		threadCount = size / blockSize;
		if (size % blockSize > 0) {
			threadCount++;
		}
		
		gpu_merge<<<1, threadCount>>>(A, B, blockSize, last);
		
		cudaDeviceSynchronize();
		
		A = (A == d_array)? d_temp : d_array;
		B = (B == d_array)? d_temp : d_array;
	}
	
	cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(d_array);
	cudaFree(d_temp);
}

int main(int argc, char* argv[]) {
    int *array, *aux;
 
    
    // These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	array =  new int[SIZE];
	random_array(array, SIZE);
	display_array("before", array);
	
	aux = new int[SIZE];

	cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
		memcpy(aux, array, sizeof(int) * SIZE);
        
		start = high_resolution_clock::now();

		merge_sort(aux, SIZE);
		
		end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
	
	display_array("after", aux);
    cout << "avg time = " << fixed << setprecision(3) 
        << (timeElapsed / N) <<  " ms\n";

    delete [] array;
    delete [] aux;

    return 0;
}