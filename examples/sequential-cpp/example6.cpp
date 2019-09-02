#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils/cheader.h"

#define SIZE	10000000
#define PI		3.14159265
#define A		0.2761711861	
#define B		0.5530207692
#define C		0.3075097067
#define K		20

struct coefficient {
	double real, img;
};

int main(int argc, char* argv[]) {
	double *function, *sine, *cosine, ms;
	struct coefficient coef[K];
	int i, j, k;
	
	function = (double*) malloc(sizeof(double) * SIZE);
	sine = (double*) malloc(sizeof(double) * SIZE);
	cosine = (double*) malloc(sizeof(double) * SIZE);
	
	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();
		
		for (j = 0; j < SIZE; j++) {
			function[j] = (((A * (double) j) + (B * (double) j)) - C);
			cosine[j] = cos((2 * j * K * PI) / N);
        	sine[j] = sin((2 * j * K * PI) / N);
		}
		
		for (j = 0; j < K; j++) {
			for (k = 0; k < SIZE; k++) {
				coef[j].real = function[k] * cosine[k];
				coef[j].img = function[k] * sine[k];
			}
		}
		ms += stop_timer();
	}
	for (j = 0; j < K; j++) {
		printf("%lf - %lfi\n", coef[j].real, coef[j].img);
	}
	printf("avg time = %.5lf ms\n", (ms / N));
	
	free(function);
	free(sine);
	free(cosine);
	
	return 0;
}
