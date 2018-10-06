/* This code implements the known sort algorithm "Counting sort" */

#include <iostream>
#include <cstring>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "utils/cppheader.h"

using namespace std;
using namespace tbb;

const int SIZE = 100000;
const int GRAIN = 10000;

class EnumerationSort {
private:
	int *myArray, *myTemp;
	int mySize;
	
public:
	EnumerationSort(int *array, int *temp, int size) 
		: myArray(array), myTemp(temp), mySize(size) {}
		
	void operator() (const blocked_range<int> &r) const {
		for (int i = r.begin(); i != r.end(); i++) {
			int count = 0;
			for (int j = 0; j < mySize; j++) {
				if (myArray[j] < myArray[i]) {
					count++;
				} else if (myArray[j] == myArray[i] && j < i) {
					count++;
				}
			}
			myTemp[count] = myArray[i];
		}
	}
};

int main(int argc, char* argv[]) {
	Timer t;
	double ms;
	
	int *array = new int[SIZE];
	int *temp = new int[SIZE];
	random_array(array, SIZE);
	display_array("before", array);

	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();
		parallel_for( blocked_range<int>(0, SIZE, GRAIN),
	              EnumerationSort(array, temp, SIZE) );
		ms += t.stop();
	}
	memcpy(array, temp, sizeof(int) * SIZE);
	display_array("after", array);
	cout << "avg time = " << (ms /N) << endl;
	
	delete [] array;
	delete [] temp;
	
	return 0;
}

