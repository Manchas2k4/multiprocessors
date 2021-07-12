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

public class Example7_Solution extends Thread {
	private static final int SIZE = 100_000_000;
	private boolean array[];
	private int start, end;

	public Example7_Solution(boolean array[], int start, int end) {
		this.array = array;
		this.start = start;
		this.end = end;
	}

	private boolean isPrime(int n) {
		for (int i = 2; i < ((int) Math.sqrt(n)); i++) {
			if (n % i == 0) {
				return false;
			}
		}
		return true;
	}

	public void run() {
		for (int i = start; i < end; i++) {
			array[i] = isPrime(i);
		}
	}

	public static void main(String args[]) {
		boolean array[] = new boolean[SIZE + 1];
		Example7_Solution threads[];
		long startTime, stopTime;
		double ms;
		int block;

		array = new boolean[SIZE];
		for (int i = 2; i < Utils.TOP_VALUE; i++) {
			array[i] = false;
			System.out.print("" + i + ", ");
		}
		System.out.println("");

		block = SIZE / Utils.MAXTHREADS;
		threads = new Example7_Solution[Utils.MAXTHREADS];

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example7_Solution(array, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example7_Solution(array, (i * block), SIZE);
				}
			}

			startTime = System.currentTimeMillis();
			for (int i = 0; i < threads.length; i++) {
				threads[i].start();
			}
			for (int i = 0; i < threads.length; i++) {
				try {
					threads[i].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			stopTime = System.currentTimeMillis();
			ms +=  (stopTime - startTime);
		}
		System.out.println("Expanding the numbers that are prime to TOP_VALUE:");
		for (int i = 2; i < Utils.TOP_VALUE; i++) {
			if (array[i]) {
				System.out.print("" + i + ", ");
			}
		}
		System.out.println("");
		System.out.printf("avg time = %.5f ms\n", (ms / Utils.N));
	}
}
