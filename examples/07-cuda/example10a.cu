// =================================================================
//
// File: example10.cu
// Author: Pedro Perez
// Description: This file implements the code that blurs a given
//				image. Uses OpenCV, to compile:
//				nvcc example8.cu `pkg-config --cflags --libs opencv`
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <cuda_runtime.h>
#include "utils.h"

#define BLUR_WINDOW 15
#define THREADS 256
#define BLOCKS	MMAX(32, ((SIZE / THREADS) + 1))

typedef unsigned char uchar;

//enum color {BLUE, GREEN, RED};

__global__ void blur(uchar *src, uchar *dest, int height, int width,
                     int step, int channels) {
	int i, j, side_pixels, cells;
	int ren, col, tmp_ren, tmp_col;
	float r, g, b;

	ren = blockIdx.x;
	col = threadIdx.x;

	side_pixels = (BLUR_WINDOW - 1) / 2;
	cells = (BLUR_WINDOW * BLUR_WINDOW);
	r = 0; g = 0; b = 0;
	for (i = -side_pixels; i <= side_pixels; i++) {
		for (j = -side_pixels; j <= side_pixels; j++) {
			tmp_ren = MIN( MAX(ren + i, 0), height - 1 );
			tmp_col = MIN( MAX(col + j, 0), width - 1);

			r += (float) src[(tmp_ren * step) + (tmp_col * channels) + RED];
			g += (float) src[(tmp_ren * step) + (tmp_col * channels) + GREEN];
			b += (float) src[(tmp_ren * step) + (tmp_col * channels) + BLUE];
		}
	}

	dest[(ren * step) + (col * channels) + RED] =  (unsigned char) (r / cells);
	dest[(ren * step) + (col * channels) + GREEN] = (unsigned char) (g / cells);
	dest[(ren * step) + (col * channels) + BLUE] = (unsigned char) (b / cells);
}

int main(int argc, char* argv[]) {
	int i;
	double acum;
  uchar *dev_src, *dev_dest;

	IplImage *src = cvLoadImage("/content/wallpaper_1920_1080.jpg", CV_LOAD_IMAGE_COLOR);
	IplImage *dest = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 3);

  long size = src->width * src->height * src->nChannels * sizeof(uchar);
  int step = src->widthStep / sizeof(uchar);

	cudaMalloc((void**) &dev_src, size);
	cudaMalloc((void**) &dev_dest, size);

	cudaMemcpy(dev_src, src->imageData, size, cudaMemcpyHostToDevice);

	acum = 0;
	printf("Starting...\n");
	for (i = 0; i < N; i++) {
		start_timer();

		blur<<<src->height, src->width>>>(dev_src, dev_dest, src->height, src->width,
                                    step, src->nChannels);

		acum += stop_timer();
	}

	cudaMemcpy(dest->imageData, dev_dest, size, cudaMemcpyDeviceToHost);

	cudaFree(dev_dest);
	cudaFree(dev_src);

	printf("avg time = %.5lf ms\n", (acum / N));

	cvSaveImage("gpu_blur.jpg", dest);

	return 0;
}
