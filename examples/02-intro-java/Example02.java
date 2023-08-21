// =================================================================
//
// File: Example02.java
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//				X within the array and replaces it with Y. The time 
//				it takes to implement this will be used as the basis 
//				for calculating the improvement obtained with parallel 
//				technologies. The time this implementation takes.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example02 {
	private static final int SIZE = 300_000_000;
	private int array[];
	private int oldElement, newElement;

	public Example02(int array[], int oldElement, int newElement) {
		this.array = array;
		this.oldElement = oldElement;
		this.newElement = newElement;
	}

	public void doTask() {
		for (int i = 0; i < array.length; i++) {
			if (array[i] == oldElement) {
				array[i] = newElement;
			}
		}
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		int aux[] = new int[SIZE];
		long startTime, stopTime;
		double elapsedTime = 0;

		for (int i = 0; i < array.length; i++) {
			array[i] = 1;
		}
		Utils.displayArray("before", array);

		elapsedTime = 0;
		System.out.printf("Starting...\n");
		for (int j = 0; j < Utils.N; j++) {
			System.arraycopy(array, 0, aux, 0, array.length);
			
			startTime = System.currentTimeMillis();

			Example02 obj = new Example02(aux, 1, -1);
			obj.doTask();

			stopTime = System.currentTimeMillis();

			elapsedTime += (stopTime - startTime);
		}
		Utils.displayArray("after", aux);
		System.out.printf("avg time = %.5f ms\n", (elapsedTime / Utils.N));
	}
}
