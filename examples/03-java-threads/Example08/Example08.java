// =================================================================
//	
// File: Example08.java
// Author: Pedro Perez
// Description: This file implements the merge sort algorithm using 
//				Java's Threads.
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Arrays;

public class Example08 extends Thread {
	private static final int SIZE = 100_000_000;
	private int A[], B[], start, end, depth;

	public Example08(int start, int end, int depth, int A[], int B[]) {
		this.start = start;
		this.end = end;
		this.depth = depth;
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

	public void run() {
		Example08 left, right;

		if (depth == 0) {
			(new MergeSort(start, end, A, B)).doTask();
		} else {
			int mid = start + ((end - start) / 2);
			
			left = new Example08(start, mid, depth - 1, A, B);
			left.start();

			right = new Example08(mid + 1, end, depth - 1, A, B);
			right.start();

			try {
				left.join(); right.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

			merge(start, mid, end);
			copyArray(start, end);
		}
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int A[] = new int[SIZE];
		int B[] = new int[SIZE];
		int depth;
		long startTime, stopTime;
		double elapsedTime;
		Example08 obj = null;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

		depth = (int) ((Math.log(Utils.MAXTHREADS) * 2) / Math.log(2));

		System.out.printf("Starting...\n");
		elapsedTime = 0;
		for (int j = 0; j < Utils.N; j++) {
			System.arraycopy(array, 0, A, 0, array.length);

			startTime = System.currentTimeMillis();

			obj = new Example08(0, A.length - 1, depth, A, B);

			obj.start();

			try {
				obj.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		Utils.displayArray("after", A);
		System.out.printf("avg time = %.5f\n", (elapsedTime / Utils.N));
	}
}