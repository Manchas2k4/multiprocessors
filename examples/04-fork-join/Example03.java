// =================================================================
//
// File: Example03.java
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector using Java's Fork-Join technology.
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example03 extends RecursiveAction {
	private static final int RENS = 25_000;
	private static final int COLS = 25_000;
	private static final int MIN = 2_000;
	private int m[], b[], c[], start, end;

	public Example03(int start, int end, int m[], int b[], int c[]) {
		this.start = start;
		this.end = end;
		this.m = m;
		this.b = b;
		this.c = c;
	}

	private void computeDirectly() {
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
			invokeAll(new Example03(start, mid, m, b, c),
					  new Example03(mid, end, m, b, c));
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double elapsedTime;
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

		System.out.printf("Starting...\n");
		elapsedTime = 0;
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			pool.invoke(new Example03(0, RENS, m, b, c));

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
