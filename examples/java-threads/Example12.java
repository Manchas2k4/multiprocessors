// =================================================================
//
// File: Example12.java
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

public class Example12 extends Thread {
	private static final int SIZE = 100_000_000;
	private static final int GRAIN = 1_000;
	private int A[], B[], start, end, numberOfThread;

	public Example12(int A[], int B[], int start, int end, int numberOfThread) {
		this.A = A;
		this.B = B;
		this.start = start;
		this.end = end;
		this.numberOfThread = numberOfThread;
	}

	private void swap(int a[], int i, int j) {
		int aux = a[i];
		a[i] = a[j];
		a[j] = aux;
	}

	private void copyArray(int low, int high) {
		int length = high - low + 1;
		System.arraycopy(B, low, A, low, length);
	}

	private void merge(int low, int mid, int high) {
		int i, j, k;

		i = low;
		j = mid + 1;
		k = low;
		while(i <= mid && j <= high){
			if(A[i] < A[j]){
				B[k] = A[i];
				i++;
			}else{
				B[k] = A[j];
				j++;
			}
			k++;
		}
		for(; j <= high; j++){
			B[k++] = A[j];
		}

		for(; i <= mid; i++){
			B[k++] = A[i];
		}
	}

	private void simpleSort(int low, int high, int size) {
		for(int i = low + 1; i < size; i++){
			for(int j = i; j > low && A[j] < A[j - 1]; j--){
				swap(A, j, j - 1);
			}
		}
	}

	private void sequentialSplit(int low, int high) {
		int  mid;

		if((high - low + 1) < GRAIN) {
			simpleSort(low, high, high - low + 1);
			return;
		}

		mid = low + ((high - low) / 2);
		sequentialSplit(low, mid);
		sequentialSplit(mid + 1, high);
		merge(low, mid, high);
		copyArray(low, high);
	}

	private void parallelSplit() {
		int  mid, count = numberOfThread;
		Example12 left, right;

		if((end - start + 1) < GRAIN) {
			simpleSort(start, end, end - start + 1);
			return;
		}

		mid = start + ((end - start) / 2);

		count++;
		left = new Example12(A, B, start, mid, count);
		left.start();

		count++;
		right = new Example12(A, B, mid + 1, end, count);
		right.start();

		try {
			left.join(); right.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		merge(start, mid, end);
		copyArray(start, end);
	}

	public void run() {
		if (numberOfThread >= Utils.MAXTHREADS) {
			sequentialSplit(start, end);
		} else {
			parallelSplit();
		}
	}

	public int[] getSortedArray() {
		return A;
	}

	public static void main(String args[]) {
		int array[], temp[];
		long startTime, stopTime;
		double ms;
		Example12 obj = null;

		array = new int[SIZE];
		temp = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("before", array);

		System.out.printf("Starting...\n");
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			obj = new Example12(Arrays.copyOf(array, array.length), temp,
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
