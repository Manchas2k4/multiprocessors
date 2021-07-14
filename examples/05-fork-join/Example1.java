// =================================================================
//
// File: Example1.java
// Author: Pedro Perez
// Description: This file contains the code that adds all the
//				elements of an integer array using Java's
//				Fork-Join.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class Example1 extends RecursiveTask<Long> {
	private static final int SIZE = 100_000_000;
	private static final int MIN = 100_000;
	private int array[], start, end;

	public Example1(int array[], int start, int end) {
		this.array = array;
		this.start = start;
		this.end = end;
	}

	protected Long computeDirectly() {
		long result = 0;
		for (int i = start; i < end; i++) {
			result += array[i];
		}
		return result;
	}

	@Override
	protected Long compute() {
		if ( (end - start) <= MIN ) {
			return computeDirectly();
		} else {
			int mid = start + ( (end - start) / 2 );
			Example1 lowerMid = new Example1(array, start, mid);
			lowerMid.fork();
			Example1 upperMid = new Example1(array, mid, end);
			return upperMid.compute() + lowerMid.join();
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime, result = 0;
		int array[];
		double ms;
		ForkJoinPool pool;

		array = new int[SIZE];
		Utils.fillArray(array);
		Utils.displayArray("array", array);

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			result = pool.invoke(new Example1(array, 0, array.length));

			stopTime = System.currentTimeMillis();
			ms += (stopTime - startTime);
		}
		System.out.printf("sum = %d\n", result);
		System.out.printf("avg time = %.5f ms\n", (ms / Utils.N));
	}
}
