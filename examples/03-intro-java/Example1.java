// =================================================================
//
// File: Example1.java
// Author: Pedro Perez
// Description: This file contains the code that adds all the
//				elements of an integer array. The time this
//				implementation takes will be used as the basis to
//				calculate the improvement obtained with parallel
//				technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example1 {
	private static final int SIZE = 100_000_000;
	private int array[];
	private long result;

	public Example1(int array[]) {
		this.array = array;
		this.result = 0;
	}

	public long getResult() {
		return result;
	}

	public void calculate() {
		result = 0;
		for (int i = 0; i < array.length; i++) {
			result += array[i];
		}
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		long startTime, stopTime;
		double acum = 0;

		Utils.fillArray(array);
		Utils.displayArray("array", array);

		Example1 e = new Example1(array);
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
