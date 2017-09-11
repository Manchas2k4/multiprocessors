#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[]) {
	#pragma omp parallel
	printf("Hola mundo!\n");
	printf("Hola dos!\n");
	return 0;
}
