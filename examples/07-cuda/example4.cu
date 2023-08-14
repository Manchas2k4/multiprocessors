// =================================================================
//
// File: example4.cu
// Author(s):
// Description: This file contains the code to count the number of
//				even numbers within an array using CUDA.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.h"

#define SIZE 1000000000
#define THREADS	256
#define BLOCKS	MMIN(32, ((SIZE / THREADS) + 1))


// implement your code
