// =================================================================
//
// File: Example02.java
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//				X within the array and replaces it with Y using 
//				Java's Threads.
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example02 extends Thread {
	private static final int SIZE = 300_000_000;
	private int array[];
	private int oldElement, newElement, start, end;

	public Example02(int start, int end, int array[], int oldElement, int newElement) {
		this.start = start;
		this.end = end;
		this.array = array;
		this.oldElement = oldElement;
		this.newElement = newElement;
	}

	public void run() {
		for (int i = start; i < end; i++) {
			if (array[i] == oldElement) {
				array[i] = newElement;
			}
		}
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int aux[] = new int[SIZE];
		long startTime, stopTime;
		double elapsedTime = 0;
		int blockSize;
		Example02 threads[];

		for (int i = 0; i < array.length; i++) {
			array[i] = 1;
		}
		Utils.displayArray("before", array);

		blockSize = SIZE / Utils.MAXTHREADS;
		threads = new Example02[Utils.MAXTHREADS];

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int j = 0; j < Utils.N; j++) {
			System.arraycopy(array, 0, aux, 0, array.length);
			
			startTime = System.currentTimeMillis();

			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = 
					new Example02((i * blockSize), ((i + 1) * blockSize), aux, 1, -1);
				} else {
					threads[i] = new Example02((i * blockSize), SIZE, aux, 1, -1);
				}
			}

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
			
			elapsedTime += (stopTime - startTime);
		}
		Utils.displayArray("after", aux);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
