// =================================================================
//
// File: Example01.java
// Author: Pedro Perez
// Description: This file implements the addition of two vectors. 
//				The time this implementation takes will be used as 
//				the basis to calculate the improvement obtained with 
//				parallel technologies.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example01 {
	private static final int SIZE = 100_000_000;
	private int a[], b[], c[];

	public Example01(int c[], int a[], int b[]) {
		this.a = a;
		this.b = b;
		this.c = c;
	}

	public void doTask() {
		for (int i = 0; i < c.length; i++) {
			c[i] = a[i] + b[i];
		}
	}

	public static void main(String args[]) {
		int a[] = new int [SIZE];
		int b[] = new int [SIZE];
		int c[] = new int [SIZE];
		long startTime, stopTime;
		double elapsedTime = 0;

		Utils.fillArray(a);
		Utils.displayArray("a", a);
		Utils.fillArray(b);
		Utils.displayArray("b", b);

		Example01 obj = new Example01(c, a, b);
		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			obj.doTask();

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
