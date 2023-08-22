// =================================================================
//
// File: Example04.java
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array using Java's 
//				Fork-Join technology.
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class Example04 extends RecursiveTask<Integer> {
	private static final int SIZE = 700_000_000;
	private static final int MIN = 10_000;
	private int array[], start, end;
	
	public Example04(int start, int end, int array[]) {
		this.start = start;
		this.end = end;
		this.array = array;
	}

	private Integer computeDirectly() {
		int result = array[start]; 
		for (int i = start; i < end; i++) {
			if (array[i] < result) {
				result = array[i];
			}
		}
		return result;
	}

	@Override
	protected Integer compute() {
		if ( (end - start) <= MIN ) {
			return computeDirectly();
		} else {
			int mid = start + ( (end - start) / 2 );
			Example04 lowerMid = new Example04(start, mid, array);
			lowerMid.fork();
			Example04 upperMid = new Example04(mid, end, array);
			return ((int) Math.min(upperMid.compute(), lowerMid.join()));
		}
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int result = 0;
		long startTime, stopTime;
		double elapsedTime;
		ForkJoinPool pool;

		Utils.fillArray(array);
		Utils.displayArray("array", array);

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			result = pool.invoke(new Example04(0, SIZE, array));

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		System.out.printf("result = %d\n", result);
		System.out.printf("avg time = %.5f\n ms", (elapsedTime / Utils.N));
	}
}
