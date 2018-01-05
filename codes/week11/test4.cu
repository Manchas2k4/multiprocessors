#include <stdio.h>
#include <cuda_runtime.h>

int main() {
	int count;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&count);
	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("Device name: %s\n", prop.name);
	}
	return 0;
}
