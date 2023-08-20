// =================================================================
//
// File: Example03.java
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector using Java's Threads.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example03 extends Thread {
	private static final int RENS = 25_000;
	private static final int COLS = 25_000;
	private int m[], b[], c[], start, end;

	public Example03(int start, int end, int m[], int b[], int c[]) {
		this.start = start;
		this.end = end;
		this.m = m;
		this.b = b;
		this.c = c;
	}

	public void run() {
		int acum;

		for (int i = start; i < end; i++) {
			acum = 0;
			for (int j = 0; j < COLS; j++) {
				acum += (m[(i * COLS) + j] * b[i]);
			}
			c[i] = acum;
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double elapsedTime;
		int blockSize;
		Example03 threads[];

		int m[] = new int[RENS * COLS];
		int b[] = new int[RENS];
		int c[] = new int[COLS];

		for (int i = 0; i < RENS; i++) {
			for (int j = 0; j < COLS; j++) {
				m[(i * COLS) + j] = (j + 1);
			}
			b[i] = 1;
		}

		blockSize = RENS / Utils.MAXTHREADS;
		threads = new Example03[Utils.MAXTHREADS];

		System.out.printf("Starting...\n");
		elapsedTime = 0;
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = 
					new Example03((i * blockSize), ((i + 1) * blockSize), m, b, c);
				} else {
					threads[i] = new Example03((i * blockSize), RENS, m, b, c);
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
