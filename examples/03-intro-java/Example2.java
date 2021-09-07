// =================================================================
//
// File: Example2.java
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array. The time this
//				implementation takes will be used as the basis to
//				calculate the improvement obtained with parallel
//				technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================


public class Example2 {
	private static final int SIZE = 100_000_000;
	private int array[];
	private int result;

	public Example2(int array[]) {
		this.array = array;
		this.result = 0;
	}

	public int getResult() {
		return result;
	}

	public void calculate() {
		result = 0;

		result = Integer.MAX_VALUE; // Float, Double, Long, Character
		for (int i = 0; i < array.length; i++) {
			result = Math.min(result, array[i]);
		}
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		long startTime, stopTime;
		double ms;

		Utils.randomArray(array);
		Utils.displayArray("array", array);

		int pos = Math.abs(Utils.r.nextInt()) % SIZE;
		System.out.printf("Setting value 0 at %d\n", pos);
		array[pos] = 0;

		Example2 e = new Example2(array);
		ms = 0;
		System.out.printf("Starting...\n");
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			e.calculate();

			stopTime = System.currentTimeMillis();

			ms += (stopTime - startTime);
		}
		System.out.printf("result = %d\n", e.getResult());
		System.out.printf("avg time = %.5f\n ms", (ms / Utils.N));
	}
}
