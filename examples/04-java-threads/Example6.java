// =================================================================
//
// File: Example6.java
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector using Java's Threads.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example6 extends Thread {
	private static final int RENS = 20_000;
	private static final int COLS = 20_000;
	private int m[], b[], c[], start, end;

	public Example6(int m[], int b[], int c[], int start, int end) {
		this.m = m;
		this.b = b;
		this.c = c;
		this.start = start;
		this.end = end;
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
		double ms;
		int block;
		Example6 threads[];

		int m[] = new int[RENS * COLS];
		int b[] = new int[RENS];
		int c[] = new int[COLS];

		for (int i = 0; i < RENS; i++) {
			for (int j = 0; j < COLS; j++) {
				m[(i * COLS) + j] = (j + 1);
			}
			b[i] = 1;
		}

		block = RENS / Utils.MAXTHREADS;
		threads = new Example6[Utils.MAXTHREADS];

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example6(m, b, c, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example6(m, b, c, (i * block), RENS);
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
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
