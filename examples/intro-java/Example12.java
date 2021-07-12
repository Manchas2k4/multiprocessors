// =================================================================
//	
// File: Example12.java
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

public class Example12 {
	private static final int SIZE = 100_000_000;
	private static final int GRAIN = 1_000;
	private int A[], B[];

	public Example12(int A[]) {
		this.A = A;
		this.B = new int[A.length];
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

	private void split(int low, int high) {
		int  mid, size, i, j;

		size = high - low + 1;
		if(size < GRAIN) {
			for(i = low + 1; i < size; i++){
				for(j = i; j > low && A[j] < A[j - 1]; j--){
					swap(A, j, j - 1);
				}
			}
			return;
		}

		mid = low + ((high - low) / 2);
		split(low, mid);
		split(mid + 1, high);
		merge(low, mid, high);
		copyArray(low, high);
	}

	public void doTask() {
		split(0, A.length - 1);
	}

	public int[] getSortedArray() {
		return A;
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		long startTime, stopTime;
		double ms;
		Example12 obj = null;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

		System.out.printf("Starting...\n");
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			obj = new Example12(Arrays.copyOf(array, array.length));
			obj.doTask();

			stopTime = System.currentTimeMillis();

			ms += (stopTime - startTime);
		}
		Utils.displayArray("after", obj.getSortedArray());
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
