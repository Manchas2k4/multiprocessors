// =================================================================
//	
// File: Example08.java
// Author: Pedro Perez
// Description: This file implements the merge sort algorithm using 
//				Java's Fork-Join technology.
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Arrays;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example08 extends RecursiveAction {
	private static final int SIZE = 100_000_000;
	private static final int MIN = 10_000;
	private int A[], B[], start, end;

	public Example08(int start, int end, int A[], int B[]) {
		this.start = start;
		this.end = end;
		this.A = A;
		this.B = B;
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

	private void computeDirectly() {
		(new MergeSort(start, end, A, B)).doTask();
	}

	@Override
	protected void compute() {
		if ( (end - start) <= MIN ) {
			computeDirectly();
		} else {
			int mid = start + ((end - start) / 2);
			invokeAll(new Example08(start, mid, A, B),
					  new Example08(mid + 1, end, A, B));
			merge(start, mid, end);
			copyArray(start, end);
		}
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int A[] = new int[SIZE];
		int B[] = new int[SIZE];
		long startTime, stopTime;
		double elapsedTime;
		ForkJoinPool pool;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

		System.out.printf("Starting...\n");
		elapsedTime = 0;
		for (int j = 0; j < Utils.N; j++) {
			System.arraycopy(array, 0, A, 0, array.length);

			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			pool.invoke(new Example08(0, SIZE - 1, A, B));

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		Utils.displayArray("after", A);
		System.out.printf("avg time = %.5f\n", (elapsedTime / Utils.N));
	}
}