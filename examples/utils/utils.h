// =================================================================
//
// File: utils.h 
// Author: Pedro Perez
// Description: This file contains the interface and implementation
//				of the Chronometer class. As well as the 
//				implementation of functions for the initialization 
//				and display of integer arrays.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.  
// All Rights Reserved. May be reproduced for any non-commercial 
// purpose.
//
// =================================================================
#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>

#define N 			10
#define DISPLAY		100
#define MAX_VALUE	10000

struct timeval startTime, stopTime;
int started = 0;

void start_timer() {
	started = 1;
	gettimeofday(&startTime, NULL);
}

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

void random_array(int *array, int size) {
	int i;

	srand(time(0));
	for (i = 0; i < size; i++) {
		array[i] = (rand() % 100) + 1;
	}
}

void fill_array(int *array, int size) {
	int i;

	for (i = 0; i < size; i++) {
		array[i] = (i % MAX_VALUE) + 1;
	}
}

void display_array(char *text, int *array) {
	int i;

	printf("%s = [%4i", text, array[0]);
	for (i = 1; i < DISPLAY; i++) {
		printf(",%4i", array[i]);
	}
	printf(", ... ,]\n");
}

#endif
