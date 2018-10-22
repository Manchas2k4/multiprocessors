#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel(void) {
	printf("GPU: Hello world\n");
}

int main(int argc, char* argv[]) {
	kernel<<<2, 4>>>();
	cudaDeviceSynchronize();

	return 0;
}
