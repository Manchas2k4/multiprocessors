// =================================================================
//
// File: example9.cu
// Author: Pedro Perez
// Description: This file implements the code  will generate a
//				fractal image. Uses OpenCV and OpenMP, to compile:
//				nvcc example9.cu -std=c++11 `pkg-config --cflags --libs opencv4`
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
//#include <cuda_runtime.h>
//#include <opencv2/highgui.hpp>
//#include <opencv2/cudaimgproc.hpp>
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

__global__ void build_julia_set(uchar *img, int height, int width, int step, int channels) {
  int ren = blockIdx.x;
  int col = threadIdx.x;

	int value = julia_value(col, ren, width, height);
  img[(ren * step) + (col * channels) + RED] = (uchar) (MAX_COLOR * RED_PCT * value);
  img[(ren * step) + (col * channels) + GREEN] = (uchar) (MAX_COLOR * GREEN_PCT * value);
  img[(ren * step) + (col * channels) + BLUE] = (uchar) (MAX_COLOR * BLUE_PCT * value);
}

int main(int argc, char* argv[]) {
  int i;
  double acum;
  cv::Mat image = cv::Mat(HEIGHT, WIDTH, CV_8UC3);
  cv::cuda::GpuMat d_image = cv::cuda::GpuMat(HEIGHT, WIDTH, CV_8UC3);

  /*
  img = image.isContinuous()? image.data : image.clone().data;
  size = image.total() * image.channels();
  step = image.channels() * image.step;

  printf("size = %li, width = %i, height = %i, step = %i, channels = %i\n",
    size, image.rows, image.cols, step, image.channels());

  cudaMalloc((void**) &d_img, size);
*/

  printf("width = %i, height = %i, step = %i, channels = %i\n", d_image.rows, d_image.cols, d_image.step, d_image.channels());

  d_image.upload(image);

  acum = 0;
  for (i = 0; i < N; i++) {
    start_timer();

    /*
    build_julia_set<<<HEIGHT, WIDTH>>>(d_img, image.rows, image.cols,
      step, image.channels());
      */
    build_julia_set<<<HEIGHT, WIDTH>>>((uchar*) d_image.data,
      d_image.cols, d_image.rows, d_image.step, d_image.channels());

    acum += stop_timer();
  }

  //cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

  //cv::Mat restored = cv::Mat(image.rows, image.cols, image.type(), img);
  printf("avg time = %.5lf ms\n", (acum / N));
  /*
  cv::namedWindow("CPU Julia | c(-0.8, 0.156)", cv::WINDOW_AUTOSIZE);
  cv::imshow("CPU Julia | c(-0.8, 0.156)", img);

  cv::waitKey(0);
  */
  //cv::imwrite("julia_set.jpg", restored);
  return 0;
}
