// =================================================================
//
// File: Exercise02.java
// Author(s):
// Description: This file contains the code to brute-force all
//				prime numbers less than MAXIMUM using Java's Threads.
//
// Reference:
// 	Read the document "exercise02.pdf"
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Exercise02 extends Thread {
	private static final int SIZE = 1_000_001;
	private int start, end;
	public double result;

	public Exercise02(int start, int end) {
		// place your code here
	}

	// place your code here

	public void run() {
		// place yout code here
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double result = 0, elapsedTime;
		int blockSize;
		Exercise02 threads[];
		
		// place yout code here

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
