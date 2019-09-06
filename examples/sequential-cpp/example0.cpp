/* This code calculates the sum of all elements in an array */
#include <iostream>
#include <cmath>
#include "utils/cppheader.h"

const int SIZE = 1000000000;

using namespace std;

class SumArray {
private:
	int *myArray, mySize;
	double result;
	
public:
	SumArray(int *array, int size) : myArray(array), mySize(size), result(0) {}
	
	double getResult() const {
		return result;
	}
	
	void calculate() {
		result = 0;
		for (int i = 0; i < mySize; i++) {
			result += myArray[i];
		}
	}
};

int main(int argc, char* argv[]) {
	double ms;
	Timer t;
	int *a;
	
	a = new int[SIZE];
	fill_array(a, SIZE);
	display_array("a", a);
	
	
	ms = 0;
	SumArray obj(a, SIZE);
	cout << "Starting..." << endl;
	for (int i = 0; i < N; i++) {
		t.start();
		obj.calculate();
		ms += t.stop();
	}
	cout << "sum = " << obj.getResult() << endl;
	cout << "avg time = " << (ms / N) << " ms" << endl;
	
	delete [] a;
	return 0;
}
