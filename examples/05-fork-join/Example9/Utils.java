// =================================================================
//
// File  : Utils.java
// Author: Pedro Perez
// Description: This file contains the implementation of the functions
//				for initializing integer arrays.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Random;

public class Utils {
	private static final int DISPLAY = 100;
	public static final int TOP_VALUE = 10_000;
	public static final Random r = new Random();

	public static final int MAXTHREADS = Runtime.getRuntime().availableProcessors();
	public static final int N = 10;

	// =================================================================
	// Initializes an array with random values between 1 and TOP_VALUE.
	//
	// @param array, an array of integer numbers.
	// @param size, the amount of numbers.
	// =================================================================
	public static void randomArray(int array[]) {
		for (int i = 0; i < array.length; i++) {
			array[i] = r.nextInt(TOP_VALUE) + 1;
		}
	}

	// =================================================================
	// Initializes an array with consecutive values of 1 and TOP_VALUE
	// across all locations.
	//
	// @param array, an array of integer numbers.
	// @param size, the amount of numbers.
	// =================================================================
	public static void fillArray(int array[]) {
		for (int i = 0; i < array.length; i++) {
			array[i] = (i % TOP_VALUE) + 1;
		}
	}
	// =================================================================
	// Displays the first N locations in the array.
	//
	// @param array, an array of integer numbers.
	// @param size, the amount of numbers.
	// =================================================================
	public static void displayArray(String text, int array[]) {
		int limit = (int) Math.min(DISPLAY, array.length);

		System.out.printf("%s = [%4d", text, array[0]);
		for (int i = 1; i < limit; i++) {
			System.out.printf(",%4d", array[i]);
		}
		System.out.printf(", ..., ]\n");
	}
}
