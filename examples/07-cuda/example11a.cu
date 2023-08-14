// =================================================================
//
// File: example11.cu
// Author(s):
// Description: This file implements the code that transforms a
//				grayscale image. Using OpenCV and OpenMP, to compile:
//				nvcc example11.cu `pkg-config --cflags --libs opencv4`
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
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <cuda_runtime.h>
#include "utils.h"

#define THREADS 256
#define BLOCKS	MMAX(32, ((SIZE / THREADS) + 1))

// implement your code

int main(int argc, char* argv[]) {
	int i;
	double acum;

	if (argc != 2) {
	printf("usage: %s source_file\n", argv[0]);
		return -1;
	}

	cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat dest = cv::Mat(src.rows, src.cols, CV_8UC3);
	if (!src.data) {
	printf("Could not load image file: %s\n", argv[1]);
		return -1;
	}

	acum = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		// call the implemented function

		acum += stop_timer();
	}

	printf("avg time = %.5lf ms\n", (acum / N));

	/*
	cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", src);

	cv::namedWindow("Gray", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gray", dest);

	cv::waitKey(0);
	*/
	cv::imwrite("gray_scale.png", dest);
