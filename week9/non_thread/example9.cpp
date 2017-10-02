#include <iostream>
#include "utils/cppheader.h"

using namespace std; 

const int SIZE = 1000000000;

class ArrayInitializer {
private:
	int *myArray;
	int mySize;
	
public:
	ArrayInitializer(int *array, int size) : myArray(array), mySize(size) {
		srand(time(0));
	}
	
	void doTask() {
		for (int i = 0; i < mySize; i++) {
			myArray[i] = (i % MAX_VALUE) + 1;
		}
	}
};

class Squares {
private:
	int *myArray;
	int mySize;
	
public:
	Squares(int *array, int size) : myArray(array), mySize(size) {
		srand(time(0));
	}
	
	void doTask() {
		for (int i = 0; i < mySize; i++) {
			myArray[i] *= myArray[i];
		}
	}
};

int main(int argc, char* argv[]) {
	Timer t;
	double ms;
	
	int *a = new int[SIZE];
	
	ArrayInitializer init(a, SIZE);
	Squares sq(a, SIZE); 
	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();
		init.doTask();
		sq.doTask();
		ms += t.stop();
	}
	display_array("a", a);
	cout << "avg time = " << (ms/N) << " ms" << endl;
	
	delete [] a;
	
	return 0;
}
