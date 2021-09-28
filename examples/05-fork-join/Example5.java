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
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example5 extends RecursiveAction {
	private static final int SIZE = 100_000;
	private static final int MIN = 1_000;
	private int array[], temp[], start, end;

	public Example5(int array[], int temp[], int start, int end) {
		this.array = array;
		this.temp = temp;
		this.start = start;
		this.end = end;
	}

	private void computeDirectly() {
		int aux;

		for (int i = end - 1; i > start; i--) {
			for (int j = 0; j < i; j++) {
				if (array[j] > array[j + 1]) {
					aux = array[j];
					array[j] = array[j + 1];
					array[j + 1] = aux;
				}
			}
		}
	}

	private void mergeAndCopy() {
		int i, j, k;
		int mid = start + ((end - start) / 2);

		i = start;
		j = mid;
		k = start;
		while (i < mid && j < end) {
			if (array[i] < array[j]) {
				temp[k] = array[i];
				i++;
			} else {
				temp[k] = array[j];
				j++;
			}
			k++;
		}
		for (; j < end; j++) {
			temp[k++] = array[j];
		}
		for (; i < mid; i++) {
			temp[k++] = array[i];
		}

		for (i = start; i < end; i++) {
			array[i] = temp[i];
		}
	}

	@Override
	protected void compute() {
		if ( (end - start) <= MIN ) {
			computeDirectly();
		} else {
			int mid = start + ((end - start) / 2);
			invokeAll(new Example5(array, temp, start, mid),
					  new Example5(array, temp, mid, end));
			mergeAndCopy();
		}
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int temp[] = new int[SIZE];
		long startTime, stopTime;
		double ms;
		ForkJoinPool pool;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			pool.invoke(new Example5(array, temp, 0, array.length));

			stopTime = System.currentTimeMillis();
			ms += (stopTime - startTime);
		}
		Utils.displayArray("after", array);
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
