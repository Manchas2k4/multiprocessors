// =================================================================
//
// File: Example6.java
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector. The time this implementation takes will
//				be used as the basis to calculate the improvement
//				obtained with parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example6 {
	private static final int RENS = 20_000;
	private static final int COLS = 20_000;
	private int m[], b[], c[];

	public Example6(int m[], int b[], int c[]) {
		this.m = m;
		this.b = b;
		this.c = c;
	}

	public void calculate() {
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
		double ms;

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
		ms = 0;
		Example6 e = new Example6(m, b, c);
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			e.calculate();

			stopTime = System.currentTimeMillis();

			ms += (stopTime - startTime);
		}
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f ms\n", (ms / Utils.N));
	}
}
