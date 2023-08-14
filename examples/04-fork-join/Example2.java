// =================================================================
//
// File: Example2.java
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array using Java's
//				Fork-Join.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class Example2 extends RecursiveTask<Integer> {
	private static final int SIZE = 100_000_000;
	private static final int MIN = 10_000;
	private int array[], start, end;

	public Example2(int array[], int start, int end) {
		this.array = array;
		this.start = start;
		this.end = end;
	}

	public Integer computeDirectly() {
		int result = Integer.MAX_VALUE;
		for (int i = start; i < end; i++) {
			result = (int) Math.min(result, array[i]);
		}
		return result;
	}

	@Override
	protected Integer compute() {
		if ( (end - start) <= MIN ) {
			return computeDirectly();
		} else {
			int mid = start + ( (end - start) / 2 );
			Example2 lowerMid = new Example2(array, start, mid);
			lowerMid.fork();
			Example2 upperMid = new Example2(array, mid, end);
			return ((int) Math.min(upperMid.compute(), lowerMid.join()));
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		int array[], result = 0;
		double ms;
		ForkJoinPool pool;

		array = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("array", array);

		int pos = Math.abs(Utils.r.nextInt()) % SIZE;
		System.out.printf("Setting value 0 at %d\n", pos);
		array[pos] = 0;

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			result = pool.invoke(new Example2(array, 0, array.length));

			stopTime = System.currentTimeMillis();
			ms += (stopTime - startTime);
		}
		System.out.printf("result = %d\n", result);
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
