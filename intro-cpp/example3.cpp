/* This code implements the known sort algorithm "Counting sort" */

#include <iostream>
#include <cstring>
#include "utils/cppheader.h"

using namespace std;

#define SIZE 10000

class EnumerationSort {
private:
	int *array;
	int size;

public:
	EnumerationSort(int *a, int s) : array(a), size(s) {}

	void doSort(int copy) {
		int *temp = new int[size];
		int i, j, count;

		for (i = 0; i < size; i++) {
			count = 0;
			for (j = 0; j < size; j++) {
				if (array[j] < array[i]) {
					count++;
				} else if (array[i] == array[j] && j < i) {
					count++;
				}
			}
			temp[count] = array[i];
		}
		if (copy) {
			memcpy(array, temp, sizeof(int) * size);
		}
		delete [] temp;
	}
};

int main(int argc, char* argv[]) {
	Timer t;
	double ms;
	
	int *array = new int[SIZE];
	random_array(array, SIZE);
	display_array("before", array);

	EnumerationSort es(array, SIZE);
	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();
		es.doSort((i % N == 0));
		ms += t.stop();
	}
	display_array("after", array);
	cout << "avg time = " << (ms /N) << endl;
	
	delete [] array;
	
	return 0;
}

