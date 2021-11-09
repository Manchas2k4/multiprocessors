// =================================================================
//
// File: utils.h
// Author: Pedro Perez
// Description: This file contains the implementation of the
//				functions used to take the time and perform the
//				speed up calculation; as well as functions for
//				initializing integer arrays.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#ifndef UTILS_H
#define UTILS_H

#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>

#define MMIN(a,b) (((a)<(b))?(a):(b))
#define MMAX(a,b) (((a)>(b))?(a):(b))

#define N 				10
#define DISPLAY		100
#define TOP_VALUE	10000

typedef unsigned char uchar;
typedef enum color {BLUE, GREEN, RED} Color;

struct timeval startTime, stopTime;
int started = 0;

// =================================================================
// Records the initial execution time.
// =================================================================
void start_timer() {
	started = 1;
	gettimeofday(&startTime, NULL);
}

// =================================================================
// Calculates the number of microseconds that have elapsed since
// the initial time.
//
// @returns the time passed
// =================================================================
double stop_timer() {
	long seconds, useconds;
	double duration = -1;

	if (started) {
		gettimeofday(&stopTime, NULL);
		seconds  = stopTime.tv_sec  - startTime.tv_sec;
		useconds = stopTime.tv_usec - startTime.tv_usec;
		duration = (seconds * 1000.0) + (useconds / 1000.0);
		started = 0;
	}
	return duration;
}

// =================================================================
// Initializes an array with random values between 1 and TOP_VALUE.
//
// @param array, an array of integer numbers.
// @param size, the amount of numbers.
// =================================================================
void random_array(int *array, int size) {
	int i;

	srand(time(0));
	for (i = 0; i < size; i++) {
		array[i] = (rand() % TOP_VALUE) + 1;
	}
}

// =================================================================
// Initializes an array with consecutive values of 1 and TOP_VALUE
// across all locations.
//
// @param array, an array of integer numbers.
// @param size, the amount of numbers.
// =================================================================
void fill_array(int *array, int size) {
	int i;

	srand(time(0));
	for (i = 0; i < size; i++) {
		array[i] = (i % TOP_VALUE) + 1;
	}
}

// =================================================================
// Displays the first N locations in the array.
//
// @param array, an array of integer numbers.
// @param size, the amount of numbers.
// =================================================================
void display_array(const char *text, int *array) {
	int i;

	printf("%s = [%4i", text, array[0]);
	for (i = 1; i < DISPLAY; i++) {
		printf(",%4i", array[i]);
	}
	printf(", ... ,]\n");
}
#endif
