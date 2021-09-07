// =================================================================
//
// File: example9.cpp
// Author: Pedro Perez
// Description: This file implements the code  will generate a
//				fractal image. Uses OpenCV, to compile:
//				g++ example9.cpp `pkg-config --cflags --libs opencv4`
//
//				The time this implementation takes will be used as the
//				basis to calculate the improvement obtained with
//				parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.h"

#define WIDTH     1920
#define HEIGHT		1080
#define SCALEX		0.500
#define SCALEY		0.500
#define N       	10
#define MAX_COLOR 255
#define RED_PCT		0.2
#define GREEN_PCT	0.4
#define BLUE_PCT	0.7

typedef struct complex {
    float real, img;
} Complex;

float magnitude2(const Complex *a) {
    return (a->real * a->real) + (a->img * a->img);
}

void mult(Complex *result, const Complex *a, const Complex *b) {
    result->real = (a->real * b->real) - (a->img * b->img);
    result->img = (a->img * b->real) + (a->real * b->img);
}

void add(Complex *result, const Complex *a, const Complex *b) {
    result->real = a->real + b->real;
    result->img = a->img + b->img;
}

class JuliaSet {
private:
	cv::Mat &img;

	int juliaValue(int x, int y, int width, int height) {
		int k;
		float jx = SCALEX * (float) (width / 2 - x) / (width / 2);
		float jy = SCALEY * (float) (height / 2 - y) / (height / 2);
		Complex c = {-0.8, 0.156};
		Complex a = {jx, jy};
		Complex aux;

		for (k = 0; k < 200; k++) {
			mult(&aux, &a, &a);
			add(&a, &aux, &c);
			if (magnitude2(&a) > 1000) {
				return 0;
			}
		}
		return 1;
	}

public:
	JuliaSet(cv::Mat &image) : img(image) {}

	void doTask() {
	int value;

	for(int i = 0; i < img.rows; i++) {
		for(int j = 0; j < img.cols; j++) {
			value = juliaValue(i, j, img.rows, img.cols);
			img.at<cv::Vec3b>(i,j)[RED] = (unsigned char) (MAX_COLOR * RED_PCT * value);
			img.at<cv::Vec3b>(i,j)[GREEN] = (unsigned char) (MAX_COLOR * GREEN_PCT * value);
			img.at<cv::Vec3b>(i,j)[BLUE] = (unsigned char) (MAX_COLOR * BLUE_PCT * value);
		}
		}
	}
};

int main(int argc, char* argv[]) {
  int i;
  double acum;
  cv::Mat img = cv::Mat(HEIGHT, WIDTH, CV_8UC3);

  acum = 0;
  for (i = 0; i < N; i++) {
    start_timer();

    JuliaSet obj(img);
	obj.doTask();

    acum += stop_timer();
  }

  printf("avg time = %.5lf ms\n", (acum / N));
  /*
  cv::namedWindow("CPU Julia", cv::WINDOW_AUTOSIZE);
  cv::imshow("CPU Julia", img);

  cv::waitKey(0);
  */
  cv::imwrite("julia_set.jpg", img);
  return 0;
}
