// =================================================================
//
// File: Example4.java
// Authors:
// Description: This file contains the code to count the number of
//				even numbers within an array using Threads.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example4 extends Thread {
	private static final int SIZE = 100_000_000;
	private int array[], start, end;
	private int result;

	public Example4(int array[], int start, int end) {
		// place your code here
	}

	public int getResult() {
		return result;
	}

	public void run() {
		// place your code here.
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		int array[], block;
		Example4 threads[];
		double ms;
		long result = 0;

		array = new int[SIZE];
		Utils.fillArray(array);
		Utils.displayArray("array", array);

		block = SIZE / Utils.MAXTHREADS;
		threads = new Example4[Utils.MAXTHREADS];

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int j = 1; j <= Utils.N; j++) {
			// create the threads here.

			startTime = System.currentTimeMillis();
			// start the threads

			// wait for the thread

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
