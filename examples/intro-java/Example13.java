// =================================================================
//
// File: Example13.java
// Author: Pedro Perez
// Description: This file implements the merge sort algorithm. The
//				time this implementation takes will be used as the
//				basis to calculate the improvement obtained with
//				parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Arrays;

public class Example13 {
	private static final int SIZE = 100_000_000;
	private int A[];

	public Example13(int A[]) {
		this.A = A;
	}

	private void swap(int a[], int i, int j) {
		int aux = a[i];
		a[i] = a[j];
		a[j] = aux;
	}

	private int findPivot(int low, int high) {
		for (int i = low + 1; i <= high; i++) {
			if (A[low] > A[i]) {
				return A[low];
			} else if (A[low] < A[i]){
				return A[i];
			}
		}
		return -1;
	}

	private int makePartition(int low, int high, int pivot) {
		int i, j;

		i = low;
		j = high;
		while (i < j) {
			swap(A, i , j);
			while (A[i] < pivot) {
				i++;
			}
			while (A[j] >= pivot) {
				j--;
			}
		}
		return i;
	}

	private void quick(int low, int high) {
		int pivot, pos;

		pivot = findPivot(low, high);
		if (pivot != -1) {
			pos = makePartition(low, high, pivot);
			quick(low, pos - 1);
			quick(pos, high);
		}
	}

	public void doTask() {
		quick(0, A.length - 1);
	}

	public int[] getSortedArray() {
		return A;
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		long startTime, stopTime;
		Example13 obj = null;
		double ms;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

		System.out.printf("Starting...\n");
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			obj = new Example13(Arrays.copyOf(array, array.length));
			obj.doTask();

			stopTime = System.currentTimeMillis();

			ms += (stopTime - startTime);
		}
		Utils.displayArray("after", obj.getSortedArray());
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
