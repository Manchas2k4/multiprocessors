// =================================================================
//
// File: Example7_Solution.java
// Author: Pedro Perez
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

public class Example7_Solution {
	private static final int SIZE = 100_000_000;
	private boolean array[];

	public Example7_Solution(boolean array[]) {
		this.array = array;
	}

	private boolean isPrime(int n) {
		for (int i = 2; i < ((int) Math.sqrt(n)); i++) {
			if (n % i == 0) {
				return false;
			}
		}
		return true;
	}

	public void calculate() {
		for (int i = 2; i < array.length; i++) {
			array[i] = isPrime(i);
		}
	}

	public static void main(String args[]) {
		boolean array[] = new boolean[SIZE + 1];
		long startTime, stopTime;
		double acum = 0;

		System.out.println("At first, neither is a prime. We will display to TOP_VALUE:");
		for (int i = 2; i < Utils.TOP_VALUE; i++) {
			array[i] = false;
			System.out.print("" + i + ", ");
		}
		System.out.println("");

		Example7_Solution e = new Example7_Solution(array);
		acum = 0;
		System.out.printf("Starting...\n");
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			e.calculate();

			stopTime = System.currentTimeMillis();

			acum += (stopTime - startTime);
		}
		System.out.println("Expanding the numbers that are prime to TOP_VALUE:");
		for (int i = 2; i < Utils.TOP_VALUE; i++) {
			if (array[i]) {
				System.out.print("" + i + ", ");
			}
		}
		System.out.println("");
		System.out.printf("avg time = %.5f ms\n", (acum / Utils.N));
	}
}
