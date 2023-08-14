// =================================================================
//
// File: intro3.cu
// Author: Pedro Perez
// Description: This file shows some of the basic CUDA directives.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

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
