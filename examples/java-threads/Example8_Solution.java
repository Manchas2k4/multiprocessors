// =================================================================
//
// File: Example8_Solution.java
// Author: Pedro Perez
// Description: This file implements the enumeration sort algorithm.
// 				The time this implementation takes will be used as the
//				basis to calculate the improvement obtained with
//				parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Arrays;

public class Example8_Solution extends Thread {
	private static final int SIZE = 100_000;
	private int array[], temp[], start, end;

	public Example8_Solution(int src[], int dst[], int start, int end) {
		this.array = src;
		this.temp = dst;
		this.start = start;
		this.end = end;
	}

	public void run() {
		int count;

		for (int i = start; i < end; i++) {
			count = 0;
			for (int j = 0; j < array.length; j++) {
				if (array[j] < array[i]) {
					count++;
				} else if (array[i] == array[j] && j < i) {
					count++;
				}
			}
			temp[count] = array[i];
		}
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int temp[] = new int[SIZE];
		long startTime, stopTime;
		double ms;
		Example8_Solution threads[];
		int block;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

		block = array.length / Utils.MAXTHREADS;
		threads = new Example8_Solution[Utils.MAXTHREADS];

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example8_Solution(array, temp, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example8_Solution(array, temp, (i * block), array.length);
				}
			}

			startTime = System.currentTimeMillis();
			for (int i = 0; i < threads.length; i++) {
				threads[i].start();
			}
			for (int i = 0; i < threads.length; i++) {
				try {
					threads[i].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			stopTime = System.currentTimeMillis();
			ms +=  (stopTime - startTime);
		}
		Utils.displayArray("after", temp);
		System.out.printf("avg time = %.5f\n ms", (ms / Utils.N));
	}
}
