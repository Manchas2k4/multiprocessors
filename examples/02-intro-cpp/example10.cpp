// =================================================================
//
// File: example10.cpp
// Author: Pedro Perez
// Description: This file implements the code that blurs a given 
//				image. Uses OpenCV, to compile:
//				g++ example10.cpp `pkg-config --cflags --libs opencv`
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

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "utils.h"

const int BLUR_WINDOW = 15;

using namespace std;
using namespace cv;

class BlurImage {
private:
	Mat &src, &dest;
	
	void blurPixel(int ren, int col) {
		int side_pixels, cells;
		int tmp_ren, tmp_col;
		float r, g, b;
		
		side_pixels = (BLUR_WINDOW - 1) / 2;
		cells = (BLUR_WINDOW * BLUR_WINDOW);
		r = 0; g = 0; b = 0;
		for (int i = -side_pixels; i <= side_pixels; i++) {
			for (int j = -side_pixels; j <= side_pixels; j++) {
				tmp_ren = MIN_VAL( MAX_VAL(ren + i, 0), src.rows - 1);
				tmp_col = MIN_VAL( MAX_VAL(col + j, 0), src.cols - 1);
				
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
	BlurImage(Mat &s, Mat &d) : src(s), dest(d) {}
	
	void doTask() {
		for(int i = 0; i < src.rows; i++) {
			for(int j = 0; j < src.cols; j++) {
				blurPixel(i, j);
			}
		}
	}
};

int main(int argc, char* argv[]) {
	double ms;
	
	if (argc != 2) {
		printf("usage: %s source_file\n", argv[0]);
		return -1;
	}
	
	Mat src = imread(argv[1], cv::IMREAD_COLOR); 
	Mat dest = Mat(src.rows, src.cols, CV_8UC3);
	if (!src.data) {
		printf("Could not load image file: %s\n", argv[1]);
		return -1;
	}
	
	cout << "Starting..." << endl;
	ms = 0;
	BlurImage obj(src, dest);
	for (int  i = 0; i < N; i++) {
		start_timer();
		
		obj.doTask();
		
		ms += stop_timer();
	}
	
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;
	
	namedWindow("Image (Original)", WINDOW_AUTOSIZE);
    imshow("Image (Original)", src);                
	
	namedWindow("Image (Blur)", WINDOW_AUTOSIZE);
    imshow("Image (Blur)", dest);
	
	waitKey(0);

	return 0;
}
