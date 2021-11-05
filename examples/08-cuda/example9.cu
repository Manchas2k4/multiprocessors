// =================================================================
//
// File: example9.cu
// Author: Pedro Perez
// Description: This file implements the code  will generate a
//				fractal image using CUDA technologie. Uses OpenCV,
//        to compile:
//				nvcc example9.cu `pkg-config --cflags --libs opencv`
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
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include "utils.h"

#define WIDTH		  1920
#define HEIGHT		1080
#define SCALEX		0.500
#define SCALEY		0.500
#define N       	10
#define MAX_COLOR 255
#define RED_PCT		0.2
#define GREEN_PCT	0.1
#define BLUE_PCT	0.7
#define THREADS 256
#define BLOCKS	MMAX(32, ((SIZE / THREADS) + 1))

typedef unsigned char uchar;


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

__device__ float julia_value(int x, int y, int width, int height) {
  int k;
	float jx = SCALEX * (float) (width / 2 - x) / (width / 2);
	float jy = SCALEY * (float) (height / 2 - y) / (height / 2);
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	for (k = 0; k < 200; k++) {
	    a = a*a + c;
	    if (a.magnitude2() > 1000) {
	        return (k / 200.0);
	    }
	}
	return 1.0;
}

__global__ void julia_set(uchar *img, int height, int width, int step, int channels) {
  int ren = blockIdx.x;
  int col = threadIdx.x;

	float value = julia_value(col, ren, width, height);
  img[(ren * step) + (col * channels) + RED] = (uchar) (MAX_COLOR * RED_PCT * value);
  img[(ren * step) + (col * channels) + GREEN] = (uchar) (MAX_COLOR * GREEN_PCT * value);
  img[(ren * step) + (col * channels) + BLUE] = (uchar) (MAX_COLOR * BLUE_PCT * value);
}


void fn(uchar *ptr) {
}

int main(int argc, char* argv[]) {
  int i, step;
  long size;
  double ms;
  uchar *d_img;
  IplImage* img=cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 3);

  size = img->width * img->height * img->nChannels * sizeof(uchar);
  step = img->widthStep / sizeof(uchar);

  cudaMalloc((void**) &d_img, size);

  //cudaMemcpy(d_img, img->imageData, size, cudaMemcpyHostToDevice);

  ms = 0;
  printf("Starting...\n");
  for (i = 0; i < N; i++) {
    start_timer();

    julia_set<<<HEIGHT, WIDTH>>>(d_img, img->height, img->width,
                                 step, img->nChannels);

    ms += stop_timer();
  }

  cudaMemcpy(img->imageData, d_img, size, cudaMemcpyDeviceToHost);

  printf("avg time = %.5lf ms\n", (ms / N));

  cvSaveImage("gpu_julia.jpg", img);

  cudaFree(d_img);
  return 0;
}
