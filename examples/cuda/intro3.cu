#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel(void) {
	printf("GPU blockIdx %i threadIdx %i: Hello world!\n", blockIdx.x, threadIdx.x);
}

int main(int argc, char* argv[]) {
	kernel<<<6, 2>>>();
	cudaDeviceSynchronize();
	
	return 0;
}
