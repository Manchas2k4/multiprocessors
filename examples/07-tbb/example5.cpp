// =================================================================
//
// File: example12.cpp
// Author: Pedro Perez
// Description: This file implements the bubble sort algorithm using
//				Intel's TBB.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <tbb/parallel_invoke.h>
#include "utils.h"

#define SIZE        10000
#define MAXTHREADS  12

using namespace std;
using namespace tbb;

class OddEvenSort {
private:
    int *arr, *temp, start, end, depth;

  void doSort() {
  	int aux;

  	for (int i = end - 1; i > start; i--) {
  	  for (int j = 0; j < i; j++) {
    		if (arr[j] > arr[j + 1]) {
    		  aux = arr[j];
    		  arr[j] = arr[j + 1];
    		  arr[j + 1] = aux;
    		}
  	  }
  	}
  }

  void copyArray(int low, int high) {
		int length = high - low + 1;
		memcpy(arr + low, temp + low, sizeof(int) * length);
	}

  void merge(int low, int mid, int high) {
		int i, j, k;

		i = low;
		j = mid + 1;
		k = low;
		while(i <= mid && j <= high){
			if(arr[i] < arr[j]){
				temp[k] = arr[i];
				i++;
			}else{
				temp[k] = arr[j];
				j++;
			}
			k++;
		}
		for(; j <= high; j++){
			temp[k++] = arr[j];
		}

		for(; i<= mid; i++){
			temp[k++] = arr[i];
		}
	}

public:
  OddEvenSort(int *a, int *t, int s, int e, int d)
    : arr(a), temp(t), start(s), end(e), depth(d) {}


	void doTask() {
		int  mid, size, i, j;

		if (depth == 0) {
      doSort();
    } else {
      mid = start + ((end - start) / 2);
      parallel_invoke (
        [=] { OddEvenSort obj(arr, temp, start, mid, depth - 1); obj.doTask();},
        [=] { OddEvenSort obj(arr, temp, mid, end, depth - 1); obj.doTask();}
      );
      merge(start, mid, end);
      //copyArray(low, high);
    }
	}
};

int main(int argc, char* argv[]) {
	int i, *a, *aux, depth;
	double ms;

	a = (int*) malloc(sizeof(int) * SIZE);
	aux = (int*) malloc(sizeof(int) * SIZE);
	random_array(a, SIZE);
	display_array("before", a);

  depth = (int)(log(MAXTHREADS) / log(2));

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		//memcpy(aux, a, sizeof(int) * SIZE);
		OddEvenSort obj(a, aux, 0, SIZE, depth);
    obj.doTask();

		ms += stop_timer();

    /*
		if (i == (N - 1)) {
			memcpy(a, aux, sizeof(int) * SIZE);
		}
    */
	}
	display_array("after", a);
	printf("avg time = %.5lf ms\n", (ms / N));

	free(a); free(aux);
	return 0;
}
