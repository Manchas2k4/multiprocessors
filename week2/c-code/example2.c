/* This code adds two vectors */

#include <stdio.h>
#include <stdlib.h>
#include "utils/cheader.h"

#define SIZE 100000000

void add_vectors(int *c, int* a, int *b, int size) {
	int i;
	
	for (i = 0; i < size; i++) {
		*(c + i) = *(a + i) + *(b + i);
	}
}

int main(int argc, char* argv[]) {
	int i, j, *a, *b, *c;
	double ms;
	long r;

	a = (int*) malloc(sizeof(int) * SIZE);
	fill_array(a, SIZE);
	display_array("a", a);
	
	b = (int*) malloc(sizeof(int) * SIZE);
	fill_array(b, SIZE);
	display_array("b", b);
	
	c = (int*) malloc(sizeof(int) * SIZE);
	
	printf("Starting...\n");
	ms = 0;
	for (j = 0; j < N; j++) {
		start_timer();
		add_vectors(c, a, b, SIZE);
		ms += stop_timer();
	}
	display_array("c", c);
	printf("avg time = %.5lf\n", (ms/N));
	
	free(a);
	free(b);
	free(c);
	return 0;
}
