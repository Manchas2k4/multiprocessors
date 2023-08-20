// =================================================================
//
// File: Example01.java
// Author: Pedro Perez
// Description: This file implements the addition of two vectors 
//				using Java's Threads.
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example01 extends Thread {
	private static final int SIZE = 100_000_000;
	private int a[], b[], c[], start, end;

	public Example01(int start, int end, int c[], int a[], int b[]) {
		this.start = start;
		this.end = end;
		this.a = a;
		this.b = b;
		this.c = c;
	}

	public void run() {
		for (int i = start; i < end; i++) {
			c[i] = a[i] + b[i];
		}
	}

	public static void main(String args[]) {
		int a[] = new int [SIZE];
		int b[] = new int [SIZE];
		int c[] = new int [SIZE];
		long startTime, stopTime;
		double elapsedTime = 0;
		int blockSize;
		Example01 threads[];

		Utils.fillArray(a);
		Utils.displayArray("a", a);
		Utils.fillArray(b);
		Utils.displayArray("b", b);

		blockSize = SIZE / Utils.MAXTHREADS;
		threads = new Example01[Utils.MAXTHREADS];

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = 
					new Example01((i * blockSize), ((i + 1) * blockSize), c, a, b);
				} else {
					threads[i] = new Example01((i * blockSize), SIZE, c, a, b);
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
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
