/* This code calculates an arctan approximation for |x| < 1 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils/cheader.h"

#define LIMIT 1e9

int main(int argc, char* argv[]) {
	double sum, ms, one, x = 0.99;
	int i, j, n;
	double tmp1, tmp2, tmp3;
	
	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();
		
		sum = 0;
		for (j = 0; j < LIMIT; j++) {
			one = (j % 2 == 0)? 1.0 : -1.0;
			n = (2 * j ) + 1;
			sum = sum + ( (one / n) * pow(x, n) );
		}
		
		ms += stop_timer();
	}
	printf("arctan(0.99)->(0.78) = %.15lf\n", sum);
	printf("avg time = %.5lf ms\n", (ms / N));
	return 0;
}
