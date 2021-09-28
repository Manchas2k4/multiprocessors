// =================================================================
//
// File: Example3.java
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval
//				using Java's Fork-Join.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
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

public class Example3 extends RecursiveTask<Double> {
	private static final int RECTS = 1_000_000_000;
	private static final int MIN = 100_000;
	private double x, dx, result;
	private int start, end;
	private Function fn;

	public Example3(double x, double dx, Function fn, int start, int end) {
		this.x = x;
		this.dx = dx;
		this.fn = fn;
		this.start = start;
		this.end = end;
	}

	protected Double computeDirectly() {
		double result = 0;
		for (int i = start; i < end; i++) {
			result += fn.eval(x + (i * dx));
		}
		return (result * dx);
	}

	@Override
	protected Double compute() {
		if ( (end - start) <= MIN ) {
			return computeDirectly();
		} else {
			int mid = start + ( (end - start) / 2 );
			Example3 lowerMid = new Example3(x, dx, fn, start, mid);
			lowerMid.fork();
			Example3 upperMid = new Example3(x, dx, fn, mid, end);
			return upperMid.compute() + lowerMid.join();
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double ms, x, dx, result = 0;
		ForkJoinPool pool;

		x = 0;
		dx = (Math.PI - 0) / RECTS;

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			result = pool.invoke(new Example3(x, dx, new Sin(), 0, RECTS));

			stopTime = System.currentTimeMillis();
			ms += (stopTime - startTime);
		}
		System.out.printf("result = %.5f\n", result);
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
