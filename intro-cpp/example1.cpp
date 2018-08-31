/* This code adds all the values of an array */

#include <iostream>
#include "utils/cppheader.h"

using namespace std;

const int SIZE = 1000000000;

class SumArray {
private:
	int *array;
	int size;
	long result;

public:
	SumArray(int *a, int s) : array(a), size(s) {}

	long getResult() const {
		return result;
	}

	void calculate() {
		result = 0;
		for (int i = 0; i < size; i++) {
			result += array[i];
		}
	}
};

int main(int argc, char* argv[]) {
	Timer t;
	double ms;
	int *array = new int[SIZE];
	
	fill_array(array, SIZE);
	display_array("array", array);

	SumArray sa(array, SIZE);
	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();
		sa.calculate();
		ms += t.stop();
	}
	cout << "sum = " << sa.getResult() << endl;
	cout << "avg time = " << (ms/N) << " ms" << endl;
	
	delete [] array;
	
	return 0;
}

