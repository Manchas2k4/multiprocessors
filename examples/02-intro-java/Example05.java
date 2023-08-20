// =================================================================
//
// File: Example05.java
// Author: Pedro Perez
// Description: This file contains the approximation of Pi using the 
//				Monte-Carlo method.The time this implementation 
//				takes will be used as the basis to calculate the 
//				improvement obtained with parallel technologies.
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

public class Example05 {
	private static final int INTERVAL = 100_000;
	private static final int NUMBER_OF_POINTS = 100_000_000;
	private int numberOfPoints;
	
	public Example05(int numberOfPoints) {
		this.numberOfPoints = numberOfPoints;
	}

	public double calculate() {
		double x, y, dist;
		int count;
		Random random;

		random = new Random();
		count = 0;
		for (int i = 0; i < numberOfPoints; i++) {
			x = (random.nextInt() % (INTERVAL + 1)) / ((double) INTERVAL);
			y = (random.nextInt() % (INTERVAL + 1)) / ((double) INTERVAL);
			dist = (x * x) + (y * y);
			if (dist <= 1) {
				count++;
			}
		}
		return ((double) (4.0 * count)) / ((double) numberOfPoints);
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double elapsedTime, result = 0;

		Example05 obj = new Example05(NUMBER_OF_POINTS);

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			result = obj.calculate();

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		System.out.printf("result = %.20f\n", result);
		System.out.printf("avg time = %.5f\n ms", (elapsedTime / Utils.N));
	}
}
