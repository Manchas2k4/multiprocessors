// =================================================================
//
// File: Example06.java
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval 
//				using Java's Fork-Join technology.
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

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

public class Example06 extends RecursiveTask<Double> {
	private static final int RECTS = 1_000_000_000;
	private static final int MIN = 100_000;
	private double x0, dx;
	private Function fn;
	private int start, end;

	public Example06(int start, int end, double x0, double dx, Function fn) {
		this.start = start;
		this.end = end;
		this.x0 = x0;
		this.dx = dx;
		this.fn = fn;
	}

	private Double computeDirectly() {
		double result = 0;
		for (int i = start; i < end; i++) {
			result += fn.eval(x0 + (i * dx));
		}
		return result;
	}

	@Override
	protected Double compute() {
		if ( (end - start) <= MIN ) {
			return computeDirectly();
		} else {
			int mid = start + ( (end - start) / 2 );
			Example06 lowerMid = new Example06(start, mid, x0, dx, fn);
			lowerMid.fork();
			Example06 upperMid = new Example06(mid, end, x0, dx, fn);
			return upperMid.compute() + lowerMid.join();
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double elapsedTime, x0, dx, result = 0;
		ForkJoinPool pool;

		x0 = 0;
		dx = Math.PI / RECTS;
		
		System.out.printf("Starting...\n");
		elapsedTime = 0;
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			result = pool.invoke(new Example06(0, RECTS, x0, dx, new Sin()));

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		result = result * dx;
		System.out.printf("result = %.5f\n", result);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
