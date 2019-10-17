/* This code implements the known sort algorithm "Counting sort" */

#include <iostream>
#include <cstring>
#include "utils/cppheader.h"
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

using namespace std;
using namespace tbb;

#define SIZE 100000
#define GRAIN 10000

class EnumerationSort {
private:
	int *array, *temp;
	int size;

public:
	EnumerationSort(int *a, int *t, int s) 
		: array(a), temp(t), size(s) {}

	void operator() 
		(const blocked_range<int> &r) const {

		for (int i = r.begin(); i != r.end(); i++) {
			int count = 0;
			for (int j = 0; j < size; j++) {
				if (array[j] < array[i]) {
					count++;
				} else if (array[i] == array[j] && j < i) {
					count++;
				}
			}
			temp[count] = array[i];
		}
	}
};

int main(int argc, char* argv[]) {
	Timer t;
	double ms;
	
	int *array = new int[SIZE];
	random_array(array, SIZE);
	display_array("before", array);

	int *temp = new int[SIZE];

	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();

		parallel_for(
			blocked_range<int>(0, SIZE, GRAIN),
			EnumerationSort(array, temp, SIZE));

		ms += t.stop();
	}

	memcpy(array, temp, sizeof(int) * SIZE);
	display_array("after", array);
	cout << "avg time = " << (ms /N) << endl;
	
	delete [] temp;
	delete [] array;
	
	return 0;
}

