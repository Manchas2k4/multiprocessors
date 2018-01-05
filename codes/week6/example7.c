#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utils/cheader.h"

#define LIMITE 100000

int main(int argc, char* argv[]) {
	int i, acum, j;
	double ms;
	
	printf("Starting...\n");
	ms = 0;
	for (j = 0; j <= N; j++) {
		start_timer();
		
		acum = 0;
		#pragma omp parallel for private(i) reduction(+:acum)
		for (i = 1; i <= LIMITE; i++) {
			acum = acum + i;
		}
		
		ms += stop_timer();
	}
	printf("acum = %i\n", acum);
	printf("avg time = %.5lf\n", (ms/N));
	
	
	
	return 0;
}
