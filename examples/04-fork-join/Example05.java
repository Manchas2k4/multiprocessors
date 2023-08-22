// =================================================================
//
// File: Example05.java
// Author: Pedro Perez
// Description: This file contains the approximation of Pi using the 
//				Monte-Carlo method using Java's Fork-Join technology.
//
// Reference:
//	https://www.geogebra.org/m/cF7RwK3H
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Random;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class Example05 extends RecursiveTask<Integer> {
	private static final int INTERVAL = 100_000;
	private static final int NUMBER_OF_POINTS = INTERVAL * INTERVAL;
	private static final int MIN = 10_000;
	private int start, end;
	
	public Example05(int start, int end) {
		this.start = start;
		this.end = end;
	}

	private Integer computeDirectly() {
		double x, y, dist;
		Random random = new Random();
		int count = 0;

		for (int i = start; i < end; i++) {
			x = (random.nextDouble() * 2) - 1;
			y = (random.nextDouble() * 2) - 1;
			dist = (x * x) + (y * y);
			if (dist <= 1) {
				count++;
			}
		}
		return count;
	}

	@Override
	protected Integer compute() {
		if ( (end - start) <= MIN ) {
			return computeDirectly();
		} else {
			int mid = start + ( (end - start) / 2 );
			Example05 lowerMid = new Example05(start, mid);
			lowerMid.fork();
			Example05 upperMid = new Example05(mid, end);
			return upperMid.compute() + lowerMid.join();
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double elapsedTime, result = 0;
		int count = 0;
		ForkJoinPool pool;

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			count = pool.invoke(new Example05(0, NUMBER_OF_POINTS));

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		result = ((double) (4.0 * count)) / ((double) NUMBER_OF_POINTS);
		System.out.printf("result = %.20f\n", result);
		System.out.printf("avg time = %.5f\n ms", (elapsedTime / Utils.N));
	}
}
