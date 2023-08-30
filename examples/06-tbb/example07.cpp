// =================================================================
//
// File: example07.cpp
// Author: Pedro Perez
// Description: This file implements the code that blurs a given
//				image using TBB. Uses OpenCV, to compile:
//				g++ -o app example07.cpp `pkg-config --cflags --libs opencv4` -I/usr/local/lib/tbb/include -L/usr/local/lib/tbb/lib/intel64/gcc4.4 -ltbb
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "utils.h"

#define BLUR_WINDOW 15

typedef enum color {BLUE, GREEN, RED} Color;

using namespace std;
using namespace std::chrono;
using namespace tbb;

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

	void operator() (const blocked_range<int> &r) const {
		for (int i = r.begin(); i != r.end(); i++) {
			for(int j = 0; j < src.cols; j++) {
				blurPixel(i, j);
			}
		}
	}
};

int main(int argc, char* argv[]) {
	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	if (argc != 2) {
		cout << "usage: " << argv[0] << " source_file\n";
		return -1;
	}

	cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat dest = cv::Mat(src.rows, src.cols, CV_8UC3);
	if (!src.data) {
		cout << "Could not load image file: " << argv[1] << "\n";
		return -1;
	}

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		BlurImage obj(src, dest);
		parallel_for(blocked_range<int>(0, src.rows),  obj);

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

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
