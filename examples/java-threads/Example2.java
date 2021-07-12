// =================================================================
//
// File: Example2.java
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array using using
//				Java's Threads.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================


public class Example2 extends Thread {
	private static final int SIZE = 100_000_000;
	private int array[], start, end;
	private int result;

	public Example2(int array[], int start, int end) {
		this.array = array;
		this.start = start;
		this.end = end;
		this.result = 0;
	}

	public int getResult() {
		return result;
	}

	public void run() {
		result = Integer.MAX_VALUE;
		for (int i = start; i < end; i++) {
			result = (int) Math.min(result, array[i]);
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		int array[], block, min = 0;
		Example2 threads[];
		double ms;

		array = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("array", array);

		int pos = Math.abs(Utils.r.nextInt()) % SIZE;
		System.out.printf("Setting value 0 at %d\n", pos);
		array[pos] = 0;

		block = SIZE / Utils.MAXTHREADS;
		threads = new Example2[Utils.MAXTHREADS];

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example2(array, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example2(array, (i * block), SIZE);
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

			if (j == Utils.N) {
				min = Integer.MAX_VALUE;
				for (int i = 0; i < threads.length; i++) {
					min = (int) Math.min(min, threads[i].getResult());
				}
			}
		}
		System.out.printf("result = %d\n", min);
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
