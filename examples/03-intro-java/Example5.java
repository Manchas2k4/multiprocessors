// =================================================================
//
// File: Example5.java
// Author: Pedro Perez
// Description: This file implements the bubble sort algorithm. The
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

public class Example5 {
	private static final int SIZE = 10_000;
	private int A[];

	public Example5(int A[]) {
		this.A = A;
	}

	private void swap(int a[], int i, int j) {
		int aux = a[i];
		a[i] = a[j];
		a[j] = aux;
	}

	public void doTask() {
		for(int i = A.length - 1; i > 0; i--){
			for(int j = 0; j < i; j++){
				if(A[j] > A[j + 1]){
					swap(A, j, j + 1);
				}
			}
		}
	}

	public int[] getSortedArray() {
		return A;
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		long startTime, stopTime;
		double ms;
		Example5 obj = null;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

		System.out.printf("Starting...\n");
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			obj = new Example5(Arrays.copyOf(array, array.length));
			obj.doTask();

			stopTime = System.currentTimeMillis();

			ms += (stopTime - startTime);
		}
		Utils.displayArray("after", obj.getSortedArray());
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
