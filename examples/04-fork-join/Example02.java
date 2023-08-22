// =================================================================
//
// File: Example02.java
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//				X within the array and replaces it with Y using 
//				Java's Fork-Join technology.
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example02 extends RecursiveAction {
	private static final int SIZE = 700_000_000;
	private static final int MIN = 10_000;
	private int array[];
	private int oldElement, newElement, start, end;

	public Example02(int start, int end, int array[], int oldElement, int newElement) {
		this.start = start;
		this.end = end;
		this.array = array;
		this.oldElement = oldElement;
		this.newElement = newElement;
	}

	private void computeDirectly() {
		for (int i = start; i < end; i++) {
			if (array[i] == oldElement) {
				array[i] = newElement;
			}
		}
	}

	@Override
	protected void compute() {
		if ( (end - start) <= MIN ) {
			computeDirectly();
		} else {
			int mid = start + ((end - start) / 2);
			invokeAll(new Example02(start, mid, array, oldElement, newElement),
					  new Example02(mid, end, array, oldElement, newElement));
		}
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		long startTime, stopTime;
		double elapsedTime = 0;
		ForkJoinPool pool;

		for (int i = 0; i < array.length; i++) {
			array[i] = 1;
		}
		Utils.displayArray("before", array);

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			pool.invoke(new Example02(0, SIZE, array, 1, -1));
			
			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		Utils.displayArray("after", array);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
