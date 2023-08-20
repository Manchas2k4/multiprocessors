// =================================================================
//
// File: Example03.java
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector. The time this implementation takes will
//				be used as the basis to calculate the improvement
//				obtained with parallel technologies.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example03 {
	private static final int RENS = 25_000;
	private static final int COLS = 25_000;
	private int m[], b[], c[];

	public Example03(int m[], int b[], int c[]) {
		this.m = m;
		this.b = b;
		this.c = c;
	}

	public void doTask() {
		int acum;

		for (int i = 0; i < RENS; i++) {
			acum = 0;
			for (int j = 0; j < COLS; j++) {
				acum += (m[(i * COLS) + j] * b[i]);
			}
			c[i] = acum;
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double elapsedTime;

		int m[] = new int[RENS * COLS];
		int b[] = new int[RENS];
		int c[] = new int[COLS];

		for (int i = 0; i < RENS; i++) {
			for (int j = 0; j < COLS; j++) {
				m[(i * COLS) + j] = (j + 1);
			}
			b[i] = 1;
		}

		System.out.printf("Starting...\n");
		elapsedTime = 0;
		Example03 obj = new Example03(m, b, c);
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			obj.doTask();

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
