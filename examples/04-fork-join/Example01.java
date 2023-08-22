// =================================================================
//
// File: Example01.java
// Author: Pedro Perez
// Description: This file implements the addition of two vectors 
//				using Java's Fork-Join technology.
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Arrays;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example01 extends RecursiveAction {
	private static final int SIZE = 100_000_000;
	private static final int MIN = 10_000;
	private int a[], b[], c[], start, end;

	public Example01(int start, int end, int c[], int a[], int b[]) {
		this.start = start;
		this.end = end;
		this.a = a;
		this.b = b;
		this.c = c;
	}

	private void computeDirectly() {
		for (int i = start; i < end; i++) {
			c[i] = a[i] + b[i];
		}
	}

	@Override
	protected void compute() {
		if ( (end - start) <= MIN ) {
			computeDirectly();
		} else {
			int mid = start + ((end - start) / 2);
			invokeAll(new Example01(start, mid, c, a, b),
					  new Example01(mid, end, c, b, a));
		}
	}

	public static void main(String args[]) {
		int a[] = new int [SIZE];
		int b[] = new int [SIZE];
		int c[] = new int [SIZE];
		long startTime, stopTime;
		double elapsedTime = 0;
		ForkJoinPool pool;

		Utils.fillArray(a);
		Utils.displayArray("a", a);
		Utils.fillArray(b);
		Utils.displayArray("b", b);

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			pool.invoke(new Example01(0, a.length, c, b, a));

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
