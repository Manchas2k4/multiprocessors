// =================================================================
//
// File: intro02.cu
// Author: Pedro Perez
// Description: This file shows some of the basic CUDA directives.
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <cstdio>
#include <cuda_runtime.h>

using namespace std;

__global__ void kernel(void) {
	printf("GPU: Hello world\n");
}

int main(int argc, char* argv[]) {
	kernel<<<2, 4>>>();
	cudaDeviceSynchronize();

	return 0;
}
