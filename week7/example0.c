/* This code calculates an approximation of PI */
#include <stdio.h>
#include <stdlib.h>
#include "utils/cheader.h"

#define NUM_RECTS	100000000

int main(int argc, char* argv[]) {
	double width, area, ms;
	double sum;
	int i, j;

	printf("Starting...\n");
	ms = 0;
	for (j = 0; j < N; j++) {
		start_timer();

		sum = 0;
		width = 1.0 / (double) NUM_RECTS;
		#pragma omp parallel for private(i) shared(width) reduction(+:sum)
		for (i = 0; i < NUM_RECTS; i++) {
			double mid = (i + 0.5) * width;
			double height = 4.0 / (1.0 + (mid * mid));
			sum += height;
		}
		area = width * sum;

		ms += stop_timer();
	}
	printf("PI = %.15lf\n", area);
	printf("avg time = %.5lf ms\n", (ms / N));
	return 0;
}
