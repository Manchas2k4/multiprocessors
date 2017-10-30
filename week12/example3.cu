#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils/cheader.h"

#define SIZE 100000

__global__ void sort(int *array, int *temp, int size) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	int count = 0;
	
	for (int j = 0; j < size; j++) {
		if (array[j] < array[i]) {
			count++;
		} else if (array[j] == array[i] && j < i) {
			count++;
		}
	}
	temp[count] = array[i];
}

int main(int argc, char* argv[]) {
	int *array;
	int *d_array, *d_temp;
	double ms;
	
	array = (int*) malloc(SIZE * sizeof(int));
	random_array(array, SIZE);
	display_array("before:", array);
	
	cudaMalloc((void**) &d_array, SIZE * sizeof(int));
	cudaMalloc((void**) &d_temp, SIZE * sizeof(int));
	
	cudaMemcpy(d_array, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	printf("Starting...\n");
	ms = 0;
	for (int i = 0; i < N; i++) {
		start_timer();
		sort<<<SIZE/128, 128>>>(d_array, d_temp, SIZE);
		ms += stop_timer();
	}
	
	cudaMemcpy(array, d_temp, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("after:", array);
	printf("avg time = %.5lf ms \n", (ms/N));
	
	cudaFree(d_array);
	cudaFree(d_temp);
	
	free(array);
	
	return 0;
}
