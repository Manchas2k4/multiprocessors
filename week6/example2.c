#include <stdio.h>
#include <omp.h>

int main(int argc, char* argv[]) {
	printf("****** Using the shared clause *********\n");
	int x=1;
	#pragma omp parallel shared(x) num_threads(2)
	{
		x++;
		printf("In the parallel x is: %d\n",x);
	}
	printf("Outside the parallel x is: %d\n",x);
	
	printf("****** Using the private clause *********\n");
	x=1;
	#pragma omp parallel private(x) num_threads(2)
	{
		x++;
		printf("In the parallel x is: %d\n",x);
	}
	printf("Outside the parallel x is: %d\n",x);
	
	printf("****** Using the firstprivate clause *********\n");
	x=1;
	#pragma omp parallel firstprivate(x) num_threads(2)
	{
		x++;
		printf("In the parallel x is: %d\n",x);
	}
	printf("Outside the parallel x is: %d\n",x);


	printf("Shared:\t -> There is only one x variable it is shared and it is accessed by both threads without mutual exclusion so the final value of x may be wrong\n");
	printf("\t -> Variables are shared by default\n");	
	printf("Private:\t-> the x variable inside the contruct is a new variable private for each thread with an undefined value\n");	
	printf("Firstprivate:\t-> the x variable inside the construct is a new variable private for every thread but it is initialized to the original variable value (1)\n"); 
	return 0;
}	
