#include <stdio.h>
#include <omp.h>

int main(int argc, char* argv[]) {
	#pragma omp parallel
	{
		printf("Hello world!\n");
		printf("¡Hola mundo!\n");
	}
	return 0;
}
