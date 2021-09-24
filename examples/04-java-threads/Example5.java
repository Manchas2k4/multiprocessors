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

public class Example5 extends Thread {
	private static final int SIZE = 10_000;
	private int array[], temp[], start, end, depth;

	public Example5(int array[], int temp[], int start, int end, int depth) {
		this.array = array;
		this.temp = temp;
		this.start = start;
		this. end = end;
		this.depth = depth;
	}

	private void doSort() {
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

	public void run() {
		if (depth == 0) {
			doSort();
		} else {
			int mid = start + ((end - start) / 2);
			Example5 left = new Example5(array, temp, start, mid, depth - 1);
			Example5 right = new Example5(array, temp, mid, end, depth - 1);

			left.start(); right.start();
			try {
				left.join(); right.join();
			} catch (InterruptedException ie) {
				ie.printStackTrace();
			}
			mergeAndCopy();
		}
	}

	public int[] getSortedArray() {
		return array;
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int temp[] = new int[SIZE];

		long startTime, stopTime;
		Example5 obj = null;
		int depth = (int)(Math.log(Utils.MAXTHREADS) / Math.log(2));
		double ms;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

		System.out.printf("Starting...\n");
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			obj = new Example5(Arrays.copyOf(array, array.length), temp, 0, array.length, depth);
			obj.start();
			try {
				obj.join();
			} catch (InterruptedException ie) {
				ie.printStackTrace();
			}

			stopTime = System.currentTimeMillis();

			ms += (stopTime - startTime);
		}
		Utils.displayArray("after", obj.getSortedArray());
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
