// =================================================================
//
// File: Example03.java
// Author: Pedro Perez
// Description: This file implements the enumeration sort algorithm 
//				using Java's Fork-Join technology.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Arrays;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Exercise03 extends RecursiveAction {
	private static final int SIZE = 100_000;
	private int array[], start, end;

	public Exercise03(int start, int end, int array[]) {
		this.array = array;
		// place your code here
	}

	// place your code here
	
	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int aux[] = new int[SIZE];
		long startTime, stopTime;
		double elapsedTime;
		ForkJoinPool pool;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

		System.out.printf("Starting...\n");
		elapsedTime = 0;
		for (int i = 0; i < Utils.N; i++) {
			System.arraycopy(array, 0, aux, 0, array.length);

			startTime = System.currentTimeMillis();

			// pace your code here.

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		Utils.displayArray("after", aux);
		System.out.printf("avg time = %.5f\n", (elapsedTime / Utils.N));
	}
}
