// =================================================================
//
// File: Example13.java
// Author: Pedro Perez
// Description: This file implements the merge sort algorithm using 
//				Java's Fork-Join.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Arrays;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example13 extends RecursiveAction {
	private static final int SIZE = 100_000_000;
	private int A[], start, end;

	public Example13(int A[], int start, int end) {
		this.A = A;
		this.start = start;
		this.end = end;
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

	@Override
	protected void compute() {
		int pivot, pos;

		pivot = findPivot(start, end);
		if (pivot != -1) {
			pos = makePartition(start, end, pivot);
			invokeAll(new Example13(A, start, pos - 1),
					  new Example13(A, pos, end));
		}
	}

	public int[] getSortedArray() {
		return A;
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		long startTime, stopTime;
		ForkJoinPool pool;
		Example13 obj = null;
		double ms;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			obj = new Example13(Arrays.copyOf(array, array.length), 0, array.length - 1);
			pool.invoke(obj);

			stopTime = System.currentTimeMillis();
			ms += (stopTime - startTime);
		}
		Utils.displayArray("after", obj.getSortedArray());
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
