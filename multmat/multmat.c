/* This code adds two vectors */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/cheader.h"

#define SIZE 800
#define MAX 10

void fill_matrix(int A[SIZE][SIZE], int val) {
	int i, j;
	
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			A[i][j] = val;
		}
	}
}

void display_matrix(char* text, int A[SIZE][SIZE]) {
	int i, j;
	
	printf("%s:\n", text);
	for (i = 0; i < MAX; i++) {
		for (j = 0; j < MAX; j++) {
			printf("%6i", A[i][j]);
		}
		printf("\n");
	}
}

void mult(int C[SIZE][SIZE], int A[SIZE][SIZE], int B[SIZE][SIZE]) {
	int i;
	
	#pragma omp parallel for private(i) shared(A, B, C)
	for (i = 0; i < SIZE; i++) {
		int j, k;
		for (j = 0; j < SIZE; j++) {
			for (k = 0; k < SIZE; k++) {
				C[i][j] = C[i][j] + (A[i][k] * B[k][j]);
			}
		}
	}
}

int main(int argc, char* argv[]) {
	int i, j, A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];
	double ms;
	long r;

	fill_matrix(A, 2);
	fill_matrix(B, 1);
	fill_matrix(C, 0);
	
	display_matrix("A", A);
	display_matrix("B", B);
	
	printf("Starting...\n");
	ms = 0;
	for (j = 0; j < N; j++) {
		start_timer();
		mult(C, A, B);
		ms += stop_timer();
	}
	display_matrix("C", C);
	printf("avg time = %.5lf\n", (ms/N));

	return 0;
}
