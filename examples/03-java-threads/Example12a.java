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

public class Example12a extends Thread {
	private static final int SIZE = 10_000_000;
	private static final int GRAIN = 100_000;
	private int A[], B[], start, end, depth;

	public Example12a(int A[], int B[], int start, int end) {
		this.A = A;
		this.B = B;
		this.start = start;
		this.end = end;
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

	private void parallelSplit() {
		int  mid;
		Example12a left, right;
		
		if((end - start) <= GRAIN) {
			Arrays.sort(A, start, end);
			return;
		}

		mid = start + ((end - start) / 2);

		left = new Example12a(A, B, start, mid);
		left.start();

		right = new Example12a(A, B, mid + 1, end);
		right.start();

		try {
			left.join(); right.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		//merge(start, mid, end);
		//copyArray(start, end);
		Arrays.sort(A, start, end);
	}

	public void run() {
		parallelSplit();
	}

	public int[] getSortedArray() {
		return A;
	}

	public static void main(String args[]) {
		int array[], temp[], depth;
		long startTime, stopTime;
		double ms;
		Example12a obj = null;

		array = new int[SIZE];
		temp = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("before", array);

		depth = ((int) (Math.log(Utils.MAXTHREADS) / Math.log(2))) + 1;

		System.out.printf("Starting...\n");
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			obj = new Example12a(Arrays.copyOf(array, array.length), 
						temp, 0, array.length - 1);
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
