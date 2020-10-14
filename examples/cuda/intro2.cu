#include <stdio.h>
#include <cuda_runtime.h>

__device__ float fx(float x, float y) {
	return x + y;
}

__global__ void kernel(void) {
	printf("res = %f\n", fx(1.0, 2.0));
}

int main(int argc, char* argv[]) {
	kernel<<<1, 1>>>();
	cudaDeviceSynchronize();

	return 0;
}
