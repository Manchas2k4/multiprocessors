// =================================================================
//
// File: Exercise02.java
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

public class Exercise02 {
	private static final int SIZE = 1_000_001;

	public Exercise02() {
	}

	public double calculate() {
		// place yout code here
		return 0.0;
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double result = 0, elapsedTime;
		
		Exercise02 obj = new Exercise02();

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			// Call yout method here.

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		System.out.printf("result = %.0f\n", result);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
