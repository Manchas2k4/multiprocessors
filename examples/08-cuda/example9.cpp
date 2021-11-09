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
#include <cuda_runtime.h>
#include "utils.h"

#define WIDTH     1920
#define HEIGHT		1080
#define SCALEX		0.500
#define SCALEY		0.500
#define MAX_COLOR 255
#define RED_PCT		0.2
#define GREEN_PCT	0.4
#define BLUE_PCT	0.7

#define THREADS	256
#define BLOCKS	MMIN(32, ((SIZE / THREADS) + 1))

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia_value(int x, int y, int width, int height) {
    int k;
    float jx = SCALEX * (float) (width / 2 - x) / (width / 2);
    float jy = SCALEY * (float) (height / 2 - y) / (height / 2);
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for (k = 0; k < 200; k++) {
      a = a*a + c;
      if (a.magnitude2() > 1000) {
        return 0;
      }
    }
    return 1;
}

__global__ void build_julia_set(cv::Mat *img) {
	int value;
  int ren = blockIdx.x;
  int col = threadIdx.x;

	for(int i = 0; i < img.rows; i++) {
		for(int j = 0; j < img.cols; j++) {
			value = julia_value(i, j, img.rows, img.cols);
			img->at<cv::Vec3b>(i,j)[RED] = (unsigned char) (MAX_COLOR * RED_PCT * value);
			img->at<cv::Vec3b>(i,j)[GREEN] = (unsigned char) (MAX_COLOR * GREEN_PCT * value);
			img->at<cv::Vec3b>(i,j)[BLUE] = (unsigned char) (MAX_COLOR * BLUE_PCT * value);
		}
	}
}

int main(int argc, char* argv[]) {
  int i;
  double acum;
  cv::Mat img = cv::Mat(HEIGHT, WIDTH, CV_8UC3);

  printf("size = %li\n", sizeof(img));
  printf("size = %li\n", sizeof(cv::Mat));

  acum = 0;
  for (i = 0; i < N; i++) {
    start_timer();

    //build_julia_set(img);

    acum += stop_timer();
  }

  printf("avg time = %.5lf ms\n", (acum / N));
  /*
  cv::namedWindow("CPU Julia | c(-0.8, 0.156)", cv::WINDOW_AUTOSIZE);
  cv::imshow("CPU Julia | c(-0.8, 0.156)", img);

  cv::waitKey(0);
  */
  cv::imwrite("julia_set.jpg", img);
  return 0;
}
