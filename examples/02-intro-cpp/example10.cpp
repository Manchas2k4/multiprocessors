// =================================================================
//
// File: example10.cpp
// Author: Pedro Perez
// Description: This file implements the code that blurs a given
//				image. Uses OpenCV, to compile:
//				g++ example10.cpp `pkg-config --cflags --libs opencv4`
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
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "utils.h"

#define BLUR_WINDOW 15

class BlurImage {
private:
	cv::Mat &src, &dest;

	void blurPixel(int ren, int col) const {
		int side_pixels, cells;
		int tmp_ren, tmp_col;
		float r, g, b;

		side_pixels = (BLUR_WINDOW - 1) / 2;
		cells = (BLUR_WINDOW * BLUR_WINDOW);
		r = 0; g = 0; b = 0;
		for (int i = -side_pixels; i <= side_pixels; i++) {
			for (int j = -side_pixels; j <= side_pixels; j++) {
				tmp_ren = MIN( MAX(ren + i, 0), src.rows - 1);
				tmp_col = MIN( MAX(col + j, 0), src.cols - 1);

				r += (float) src.at<cv::Vec3b>(tmp_ren, tmp_col)[RED];
				g += (float) src.at<cv::Vec3b>(tmp_ren, tmp_col)[GREEN];
				b += (float) src.at<cv::Vec3b>(tmp_ren, tmp_col)[BLUE];
			}
		}

		dest.at<cv::Vec3b>(ren, col)[RED] =  (unsigned char) (r / cells);
		dest.at<cv::Vec3b>(ren, col)[GREEN] = (unsigned char) (g / cells);
		dest.at<cv::Vec3b>(ren, col)[BLUE] = (unsigned char) (b / cells);
	}

public:
	BlurImage(cv::Mat &s, cv::Mat &d) : src(s), dest(d) {}

	void doTask() {
		for(int i = 0; i < src.rows; i++) {
			for(int j = 0; j < src.cols; j++) {
				blurPixel(i, j);
			}
		}
	}
};

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

		BlurImage obj(src, dest);
		obj.doTask();

		acum += stop_timer();
	}

	printf("avg time = %.5lf ms\n", (acum / N));

	/*
	cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original", src);

	cv::namedWindow("Blur", cv::WINDOW_AUTOSIZE);
	cv::imshow("Blur", dest);

	cv::waitKey(0);
	*/

	cv::imwrite("blur.png", dest);

	return 0;
}
