// =================================================================
//
// File: Example4.java
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

public class Example4 {
	private static final int SIZE = 100_000_000;
	private int array[];
	private int result;

	public Example4(int array[]) {
		this.array = array;
		this.result = 0;
	}

	public int getResult() {
		return result;
	}

	public void calculate() {
		result = 0;
		// place your code here
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		long startTime, stopTime;
		double acum = 0;

		Utils.fillArray(array);
		Utils.displayArray("array", array);

		Example4 e = new Example4(array);
		acum = 0;
		System.out.printf("Starting...\n");
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			e.calculate();

			stopTime = System.currentTimeMillis();

			acum += (stopTime - startTime);
		}
		System.out.printf("sum = %d\n", e.getResult());
		System.out.printf("avg time = %.5f ms\n", (acum / Utils.N));
	}
}
