#include <stdio.h>

int main(int argc, char* argv[]) {
	int i, count;
	cudaDeviceProp prop;
	
	cudaGetDeviceCount(&count);
	for (i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("Device name: %s\n", prop.name);
	}
	return 0;
}
