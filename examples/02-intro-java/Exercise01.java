// =================================================================
//
// File: Exercise01.java
// Authors: 
// Description: This file contains the code to count the number of
//				even numbers within an array. The time this implementation
//				takes will be used as the basis to calculate the
//				improvement obtained with parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Exercise01 {
	private static final int SIZE = 100_000_000;
	private int array[];

	public Exercise01(int array[]) {
		this.array = array;
	}

	public int calculate() {
		// place your code here
		return 0;
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int result = 0;
		long startTime, stopTime;
		double elapsedTime = 0;

		Utils.fillArray(array);
		Utils.displayArray("array", array);

		Exercise01 obj = new Exercise01(array);
		
		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			result = obj.calculate();

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		System.out.printf("result = %d\n", result);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
