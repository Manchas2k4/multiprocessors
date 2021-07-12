// =================================================================
//
// File: Example3.java
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval
//				using Threads.
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

public class Example3 extends Thread {
	private static final int RECTS = 1_000_000_000;
	private double x, dx, result;
	private int start, end;
	private Function fn;

	public Example3(double a, double b, Function fn, int start, int end) {
		this.x = Math.min(a, b);
		this.dx = (Math.max(a,b) - Math.min(a, b)) / RECTS;
		this.fn = fn;
		this.start = start;
		this.end = end;
	}

	public double getResult() {
		return result;
	}

	public void run() {
		result = 0;
		for (int i = start; i < end; i++) {
			result += fn.eval(x + (i * dx));
		}
		result = result * dx;
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		int block;
		Example3 threads[];
		double ms, result = 0;

		block = RECTS / Utils.MAXTHREADS;
		threads = new Example3[Utils.MAXTHREADS];

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example3(0, Math.PI, new Sin(), (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example3(0, Math.PI, new Sin(), (i * block), RECTS);
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

			if (j == Utils.N) {
				result = 0;
				for (int i = 0; i < threads.length; i++) {
					result += threads[i].getResult();
				}
			}
		}
		System.out.printf("result = %.5f\n", result);
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
