#include <stdio.h>
#include <cuda_runtime.h>

__device__ float fx(float a, float b) {
	return a + b;
}

__global__ void kernel(void) {
	printf("result = %f\n", fx(1.0, 2.0));
}

int main(int argc, char* argv[]) {
	kernel<<<2, 2>>>();
	cudaDeviceSynchronize();
	printf("Ending...\n");
	return 0;
}
