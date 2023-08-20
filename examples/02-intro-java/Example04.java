// =================================================================
//
// File: Example04.java
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array. The time this
//				implementation takes will be used as the basis to
//				calculate the improvement obtained with parallel
//				technologies.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================


public class Example04 {
	private static final int SIZE = 1_000_000_000;
	private int array[];
	
	public Example04(int array[]) {
		this.array = array;
	}

	public int calculate() {
		int result = 0;

		result = array[0]; 
		for (int i = 1; i < array.length; i++) {
			if (array[i] < result) {
				result = array[i];
			}
		}
		return result;
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int result = 0;
		long startTime, stopTime;
		double elapsedTime;

		Utils.randomArray(array);
		Utils.displayArray("array", array);

		Example04 obj = new Example04(array);

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			result = obj.calculate();

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		System.out.printf("result = %d\n", result);
		System.out.printf("avg time = %.5f\n ms", (elapsedTime / Utils.N));
	}
}
