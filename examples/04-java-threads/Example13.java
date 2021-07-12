// =================================================================
//
// File: Example13.java
// Author: Pedro Perez
// Description: This file implements the merge sort algorithm using
//				Threads.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Arrays;

public class Example13 extends Thread {
	private static final int SIZE = 100_000_000;
	private int A[], start, end, numberOfThread;

	public Example13(int A[], int start, int end, int numberOfThread) {
		this.A = A;
		this.start = start;
		this.end = end;
		this.numberOfThread = numberOfThread;
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

	private void sequentialQuick(int low, int high) {
		int pivot, pos;

		pivot = findPivot(low, high);
		if (pivot != -1) {
			pos = makePartition(low, high, pivot);
			sequentialQuick(low, pos - 1);
			sequentialQuick(pos, high);
		}
	}

	private void parallelQuick() {
		int pivot, pos, count = numberOfThread;
		Example13 lesser, greater;

		pivot = findPivot(start, end);
		if (pivot != -1) {
			pos = makePartition(start, end, pivot);

			count++;
			lesser = new Example13(A, start, pos - 1, count);
			lesser.start();

			count++;
			greater = new Example13(A, pos, end, count);
			greater.start();

			try {
				lesser.join(); greater.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	public void run() {
		if (numberOfThread >= Utils.MAXTHREADS) {
			sequentialQuick(start, end);
		} else {
			parallelQuick();
		}
	}

	public int[] getSortedArray() {
		return A;
	}

	public static void main(String args[]) {
		int array[];
		long startTime, stopTime;
		double ms;
		Example13 obj = null;

		array = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("before", array);

		System.out.printf("Starting...\n");
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			obj = new Example13(Arrays.copyOf(array, array.length),
									0, array.length - 1, 1);
			obj.start();

			try {
				obj.join();
			} catch(InterruptedException e) {
				e.printStackTrace();
			}

			stopTime = System.currentTimeMillis();

			ms += (stopTime - startTime);
		}
		Utils.displayArray("after", obj.getSortedArray());
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
