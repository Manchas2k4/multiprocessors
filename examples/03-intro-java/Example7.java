// =================================================================
//
// File: Example7.java
// Author(s):
// Description: This file contains the code to brute-force all
//				prime numbers less than MAXIMUM. The time this
//				implementation takes will be used as the basis to
//				calculate the improvement obtained with parallel
//				technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example7 {
	private static final int SIZE = 1_000_001;
	private boolean array[];

	public Example7() {
		this.array = array;
	}

	// place yout code here

	public void calculate() {
		// place yout code here
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double result, ms;
		Example7 obj;

		ms = 0;
		result = 0;
		obj = new Example7();
		System.out.printf("Starting...\n");
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			// Call yout method here.

			stopTime = System.currentTimeMillis();

			ms += (stopTime - startTime);
		}
		System.out.printf("sum = %.0f\n", result);
		System.out.printf("avg time = %.5f ms\n", (ms / Utils.N));
	}
}
