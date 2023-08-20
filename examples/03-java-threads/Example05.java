// =================================================================
//
// File: Example05.java
// Author: Pedro Perez
// Description: This file contains the approximation of Pi using the 
//				Monte-Carlo method using Java's Threads.
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

public class Example05 extends Thread {
	private static final int INTERVAL = 100_000;
	private static final int NUMBER_OF_POINTS = 100_000_000;
	private int start, end;
	public int count;
	
	public Example05(int start, int end) {
		this.start = start;
		this.end = end;
		this.count = 0;
	}

	public void run() {
		double x, y, dist;
		Random random = new Random();

		count = 0;
		for (int i = start; i < end; i++) {
			x = (random.nextInt() % (INTERVAL + 1)) / ((double) INTERVAL);
			y = (random.nextInt() % (INTERVAL + 1)) / ((double) INTERVAL);
			dist = (x * x) + (y * y);
			if (dist <= 1) {
				count++;
			}
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double elapsedTime, result = 0;
		int blockSize, count = 0;
		Example05 threads[];

		blockSize = NUMBER_OF_POINTS / Utils.MAXTHREADS;
		threads = new Example05[Utils.MAXTHREADS];

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = 
					new Example05((i * blockSize), ((i + 1) * blockSize));
				} else {
					threads[i] = new Example05((i * blockSize), NUMBER_OF_POINTS);
				}
			}

			for (int i = 0; i < threads.length; i++) {
				threads[i].start();
			}
			
			count = 0;
			for (int i = 0; i < threads.length; i++) {
				try {
					threads[i].join();
					count += threads[i].count;
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		result = ((double) (4.0 * count)) / ((double) NUMBER_OF_POINTS);
		System.out.printf("result = %.20f\n", result);
		System.out.printf("avg time = %.5f\n ms", (elapsedTime / Utils.N));
	}
}
