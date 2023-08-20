// =================================================================
//
// File: Example06.java
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval.
//				The time this implementation takes will be used as
//				the basis to calculate the improvement obtained with
//				parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

class Sin implements Function {
	public double eval(double x) {
		return Math.sin(x);
	}
}

class Cos implements Function {
	public double eval(double x) {
		return Math.cos(x);
	}
}

public class Example06 {
	private static final int RECTS = 1_000_000_000;
	private double x0, dx, result;
	private Function fn;

	public Example06(double x0, double dx, Function fn) {
		this.x0 = x0;
		this.dx = dx;
		this.fn = fn;
	}

	public double calculate() {
		double result = 0;
		for (int i = 0; i < RECTS; i++) {
			result += fn.eval(x0 + (i * dx));
		}
		result = result * dx;
		return result;
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double elapsedTime, x0, dx, result = 0;

		x0 = 0;
		dx = Math.PI / RECTS;
		Example06 obj = new Example06(x0, dx, new Sin());

		System.out.printf("Starting...\n");
		elapsedTime = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			result = obj.calculate();

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		System.out.printf("result = %.5f\n", result);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
