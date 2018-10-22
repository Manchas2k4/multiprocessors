#include <stdio.h>
#include "utils/cheader.h"

#define SIZE 	1e5
#define THREADS 128

__global__ void sort(int *array, int *temp, int size) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	int j, count;
	
	if (i < size) {
		count = 0;
		for (j = 0; j < size; j++) {
			if (array[j] < array[i]) {
				count++;
			} else if (array[j] == array[i] && j < i) {
				count++;
			}
		}
		temp[count] = array[i];
	}
}

int main(int argc, char* argv[]) {
	int *array;
	int *d_array, *d_temp;
	double ms;
	int i;
	
	array = (int*) malloc(SIZE * sizeof(int));
	random_array(array, SIZE);
	display_array("before", array);
	
	cudaMalloc((void**) &d_array, SIZE * sizeof(int));
	cudaMalloc((void**) &d_temp, SIZE * sizeof(int));
	
	cudaMemcpy(d_array, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();
		sort<<<SIZE/THREADS, THREADS>>>(d_array, d_temp, SIZE);
		ms += stop_timer();
	}
	
	cudaMemcpy(array, d_temp, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("after", array);
	printf("avg time = %.15lf ms\n", (ms / N));
	
	cudaFree(d_array);
	cudaFree(d_temp);
	
	free(array);
	
	return 0;
}
