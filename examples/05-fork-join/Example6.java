// =================================================================
//
// File: Example6.java
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector using Java's Fork-Join.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example6 extends RecursiveAction {
	private static final int RENS = 20_000;
	private static final int COLS = 20_000;
	private static final int MIN = 5_000;
	private int m[], b[], c[], start, end;

	public Example6(int m[], int b[], int c[], int start, int end) {
		this.m = m;
		this.b = b;
		this.c = c;
		this.start = start;
		this.end = end;
	}

	public void computeDirectly() {
		int acum;

		for (int i = start; i < end; i++) {
			acum = 0;
			for (int j = 0; j < COLS; j++) {
				acum += (m[(i * COLS) + j] * b[i]);
			}
			c[i] = acum;
		}
	}

	@Override
	protected void compute() {
		if ( (end - start) <= MIN ) {
			computeDirectly();
		} else {
			int mid = start + ((end - start) / 2);
			invokeAll(new Example6(m, b, c, start, mid),
					  new Example6(m, b, c, mid, end));
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double ms;
		ForkJoinPool pool;

		int m[] = new int[RENS * COLS];
		int b[] = new int[RENS];
		int c[] = new int[COLS];

		for (int i = 0; i < RENS; i++) {
			for (int j = 0; j < COLS; j++) {
				m[(i * COLS) + j] = (j + 1);
			}
			b[i] = 1;
		}

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			pool.invoke(new Example6(m, b, c, 0, RENS));

			stopTime = System.currentTimeMillis();
			ms += (stopTime - startTime);
		}
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
