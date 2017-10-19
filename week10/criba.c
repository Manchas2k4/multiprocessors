#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SIZE 100000000

int main(int argc, char* argv[]) {
	int i, j;
	int limit = (int) floor(sqrt(SIZE));
	char *a = (char*) malloc(sizeof(char) * SIZE);
	
	memset(a, 1, sizeof(char) * SIZE);
	
	for (i = 2; i < limit; i++) {
		for (j = 2*i; j < SIZE; j += i) {
			a[j] = 0;
		}
	}
	
	printf("PRIME NUMBERS:\n");
	for (i = 2; i < SIZE; i++) {
		if (a[i]) {
			printf("%i ", i);
		}
	}
	
	free(a);
	return 0;
}
