// =================================================================
//
// File: Example06.java
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval 
//				using Java's Threads.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
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

public class Example06 extends Thread {
	private static final int RECTS = 1_000_000_000;
	private double x0, dx, result;
	private Function fn;
	private int start, end;

	public Example06(int start, int end, double x0, double dx, Function fn) {
		this.start = start;
		this.end = end;
		this.x0 = x0;
		this.dx = dx;
		this.fn = fn;
	}

	public void run() {
		result = 0;
		for (int i = start; i < end; i++) {
			result += fn.eval(x0 + (i * dx));
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double elapsedTime, x0, dx, result = 0;
		int blockSize;
		Example06 threads[];

		x0 = 0;
		dx = Math.PI / RECTS;
		
		blockSize = RECTS / Utils.MAXTHREADS;
		threads = new Example06[Utils.MAXTHREADS];

		System.out.printf("Starting...\n");
		elapsedTime = 0;
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = 
					new Example06((i * blockSize), ((i + 1) * blockSize), x0, dx, new Sin());
				} else {
					threads[i] = new Example06((i * blockSize), RECTS, x0, dx, new Sin());
				}
			}

			for (int i = 0; i < threads.length; i++) {
				threads[i].start();
			}
			
			result = 0;
			for (int i = 0; i < threads.length; i++) {
				try {
					threads[i].join();
					result += threads[i].result;
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		result = result * dx;
		System.out.printf("result = %.5f\n", result);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
