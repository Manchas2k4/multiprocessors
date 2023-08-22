// =================================================================
//
// File: Exercise02.java
// Author(s):
// Description: This file contains the code to brute-force all
//				prime numbers less than MAXIMUM using Java's 
//				Fork-Join technology.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class Exercise02 extends RecursiveTask<Long> {
	private static final int SIZE = 1_000_001;
	private static final int MIN = 10_000;
	private int start, end;

	public Exercise02(int start, int end) {
		// place your code here
	}

	// place your code here

	public static void main(String args[]) {
		long startTime, stopTime;
		double result = 0, elapsedTime;
		int blockSize;
		ForkJoinPool pool;
		
		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			// place yout code here

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		System.out.printf("result = %.0f\n", result);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
