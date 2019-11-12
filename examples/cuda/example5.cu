/* Implementation of the counting sort algorithm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/cheader.h"

#define MIN(a,b) (a<b?a:b)
#define MAX(a,b) (a>b?a:b)

#define SIZE	1e5
#define THREADS	256
#define BLOCKS	MAX(32, (SIZE + THREADS - 1)/ THREADS)

__global__ void sort(int *array, int *temp) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	int j, count;
	
	if (i < SIZE) {
		count = 0;
		for (j = 0; j < SIZE; j++) {
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
	int i, *a;
	int *da, *dt;
	double ms;
	
	a = (int*) malloc(sizeof(int) * SIZE);
	random_array(a, SIZE);
	display_array("before", a);
	
	cudaMalloc((void**) &da, SIZE * sizeof(int));
	cudaMalloc((void**) &dt, SIZE * sizeof(int));
	
	cudaMemcpy(da, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	printf("Starting...\n");
	for (i = 0; i < N; i++) {
		start_timer();
		
		sort<<<BLOCKS, THREADS>>>(da, dt);
		
		ms += stop_timer();
	}
	
	cudaMemcpy(a, dt, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	
	display_array("after", a);
	printf("avg time = %.5lf ms\n", (ms / N));
	
	cudaFree(da);
	cudaFree(dt);
	
	free(a);
	return 0;
}
