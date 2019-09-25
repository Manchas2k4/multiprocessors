/* This code calculates the sum of all elements in an array */
#include <iostream>
#include <cmath>
#include "utils/cppheader.h"

const int SIZE = 1000000000;

using namespace std;

class Deviation {
private:
	int *myArray, mySize;
	double result;
	
public:
	Deviation(int *array, int size) 
		: myArray(array), mySize(size), result(0) {}
	
	double getResult() const {
		return result;
	}
	
	void calculate() {
		int i;
		double acum, avg;
		
		acum = 0;
		for (i = 0; i < mySize; i++) {
			acum += myArray[i];
		}
		avg = acum / mySize;
		
		acum = 0;
		for (i = 0; i < mySize; i++) {
			acum += (myArray[i] - avg) * (myArray[i] - avg);
		}
		result = sqrt(acum / mySize);
	}
};

int main(int argc, char* argv[]) {
	double ms;
	Timer t;
	int *a;
	
	a = new int[SIZE];
	random_array(a, SIZE);
	display_array("a", a);
	
	
	ms = 0;
	Deviation obj(a, SIZE);
	cout << "Starting..." << endl;
	for (int i = 0; i < N; i++) {
		t.start();
		obj.calculate();
		ms += t.stop();
	}
	cout << "S = " << obj.getResult() << endl;
	cout << "avg time = " << (ms / N) << " ms" << endl;
	
	delete [] a;
	return 0;
}
