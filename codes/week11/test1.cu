#include <stdio.h>
#include <cuda_runtime.h>

__global__ void myKernel(void) {
	printf("GPU: Hello world!\n");
}


int main() {
	myKernel<<<2, 1>>>();
	cudaDeviceSynchronize();
	printf("Ending...\n");
	return 0;
}
