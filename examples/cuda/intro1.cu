#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int *c, int *a, int *b) {
	*c = *a + *b;
}

int main(int argc, char* argv[]) {
	int a, b, c;
	int *da, *db, *dc;
	
	cudaMalloc((void**) &da, sizeof(int));
	cudaMalloc((void**) &db, sizeof(int));
	cudaMalloc((void**) &dc, sizeof(int));
	
	scanf("%i %i", &a, &b);
	
	cudaMemcpy(da, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(db, &b, sizeof(int), cudaMemcpyHostToDevice);
	
	add<<<1, 1>>>(dc, da, db);
	
	cudaMemcpy(&c, dc, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("c = %i\n", c);
	
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	
	return 0;
}
