// =================================================================
//
// File: Example04.java
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array using Java's 
//				Threads.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================


public class Example04 extends Thread {
	private static final int SIZE = 770_000_000;
	private int array[], start, end;
	public int result;
	
	public Example04(int start, int end, int array[]) {
		this.start = start;
		this.end = end;
		this.array = array;
		this.result = 0;
	}

	public void run() {
		result = array[start]; 
		for (int i = start; i < end; i++) {
			if (array[i] < result) {
				result = array[i];
			}
		}
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int result = 0;
		long startTime, stopTime;
		double elapsedTime;
		int blockSize;
		Example04 threads[];

		Utils.fillArray(array);
		Utils.displayArray("array", array);

		blockSize = SIZE / Utils.MAXTHREADS;
		threads = new Example04[Utils.MAXTHREADS];

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = 
					new Example04((i * blockSize), ((i + 1) * blockSize), array);
				} else {
					threads[i] = new Example04((i * blockSize), SIZE, array);
				}
			}

			for (int i = 0; i < threads.length; i++) {
				threads[i].start();
			}
			
			result = Utils.TOP_VALUE;
			for (int i = 0; i < threads.length; i++) {
				try {
					threads[i].join();
					if (threads[i].result < result) {
						result = threads[i].result;
					}
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		System.out.printf("result = %d\n", result);
		System.out.printf("avg time = %.5f\n ms", (elapsedTime / Utils.N));
	}
}
