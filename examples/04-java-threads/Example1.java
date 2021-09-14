// =================================================================
//
// File: Example1.java
// Author: Pedro Perez
// Description: This file contains the code that adds all the
//				elements of an integer array using Java's Threads.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example1 extends Thread {
	private static final int SIZE = 100_000_000;
	private int array[], start, end;
	private long result;

	public Example1(int array[], int start, int end) {
		this.array = array;
		this.start = start;
		this.end = end;
		this.result = 0;
	}

	public long getResult() {
		return result;
	}

	public void run() {
		result = 0;
		for (int i = start; i < end; i++) {
			result += array[i];
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		int array[], block;
		Example1 threads[];
		double ms;
		long result = 0;

		array = new int[SIZE];
		Utils.fillArray(array);
		Utils.displayArray("array", array);

		block = SIZE / Utils.MAXTHREADS;
		threads = new Example1[Utils.MAXTHREADS];

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example1(array, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example1(array, (i * block), SIZE);
				}
			}

			startTime = System.currentTimeMillis();
			for (int i = 0; i < threads.length; i++) {
				threads[i].start();
			}
			/** -------- */
			for (int i = 0; i < threads.length; i++) {
				try {
					threads[i].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			stopTime = System.currentTimeMillis();
			ms +=  (stopTime - startTime);

			if (j == Utils.N) {
				result = 0;
				for (int i = 0; i < threads.length; i++) {
					result += threads[i].getResult();
				}
			}
		}
		System.out.printf("sum = %d\n", result);
		System.out.printf("avg time = %.5f ms\n", (ms / Utils.N));
	}
}
