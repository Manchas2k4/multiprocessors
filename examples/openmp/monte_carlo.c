#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils/cheader.h"
#include <omp.h>

#define NUMBER_OF_TOSSES 1000000000 //1e9

double monte_carlo() {
	double x,y;
	double min = -1, max = 1;
	double distance_squared;
	double pi_estimate;
	int i, number_in_circle = 0;
	
	#pragma omp parallel for shared(min, max) private(x, y, distance_squared) reduction(+:number_in_circle)
	for (i = 0; i < NUMBER_OF_TOSSES; i++) {
		x = min + (((double) rand() / RAND_MAX) * (max - min));
		y = min + (((double) rand() / RAND_MAX) * (max - min));
		distance_squared = (x * x) + (y * y);
		if (distance_squared <= 1) {
			number_in_circle = number_in_circle + 1;
		}
	}
	pi_estimate = 4 * number_in_circle /((double) NUMBER_OF_TOSSES);
	return pi_estimate;
}
	
int main(int argc, char* argv[]) {
	double pi_estimate, ms;
	int i;
	
	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		pi_estimate = monte_carlo();

		ms += stop_timer();
	}
	printf("PI = %.15lf\n", pi_estimate);
	printf("avg time = %.5lf ms\n", (ms / N));
	return 0;
}
