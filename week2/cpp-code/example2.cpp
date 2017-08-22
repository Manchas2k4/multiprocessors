/* This code adds two vectors */

#include <iostream>
#include "utils/cppheader.h"

using namespace std;

const int SIZE = 100000000;

class AddVectors {
private:
	int *a, *b, *c;
	int size;

public:
	AddVectors(int *arrayC, int *arrayA, int *arrayB, int s) 
		: c(arrayC), a(arrayA), b(arrayB), size(s) {}

	void calculate() {
		for (int i = 0; i < size; i++) {
			c[i] = a[i] + b[i];
		}
	}
};

int main(int argc, char* argv[]) {
	Timer t;
	double ms;
	
	int *a = new int[SIZE];
	fill_array(a, SIZE);
	display_array("a", a);
	
	int *b = new int[SIZE];
	fill_array(b, SIZE);
	display_array("b", b);
	
	int *c = new int[SIZE];
	fill_array(c, SIZE);

	AddVectors av(c, a, b, SIZE);
	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();
		av.calculate();
		ms += t.stop();
	}
	display_array("c", c);
	cout << "avg time = " << (ms/N) << " ms" << endl;
	
	delete [] a;
	delete [] b;
	delete [] c;
	
	return 0;
}

