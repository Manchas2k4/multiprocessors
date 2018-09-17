/* This code will generate a fractal image. Uses OpenCV, to compile:
   gcc example5.c `pkg-config --cflags --libs opencv`  */
#include <stdio.h>
#include <stdlib.h>
#include <opencv/highgui.h>
#include "utils/cheader.h"

#define BLUR_WINDOW 15

typedef enum color {BLUE, GREEN, RED} Color;

void blur_pixel(IplImage *src, IplImage *dest, int ren, int col) {
	int side_pixels, i, j, cells;
	int tmp_ren, tmp_col, step;
	float r, g, b;
	
	side_pixels = (BLUR_WINDOW - 1) / 2;
	cells = (BLUR_WINDOW * BLUR_WINDOW);
	step = src->widthStep / sizeof(uchar);
	r = 0; g = 0; b = 0;
	for (i = -side_pixels; i <= side_pixels; i++) {
		for (j = -side_pixels; j <= side_pixels; j++) {
			tmp_ren = MIN( MAX(ren + i, 0), src->height - 1 );
			tmp_col = MIN( MAX(col + j, 0), src->width - 1);
			
			r += (float) src->imageData[(tmp_ren * step) + (tmp_col * src->nChannels) + RED];
			g += (float) src->imageData[(tmp_ren * step) + (tmp_col * src->nChannels) + GREEN];
			b += (float) src->imageData[(tmp_ren * step) + (tmp_col * src->nChannels) + BLUE];
		}
	}
	
	dest->imageData[(ren * step) + (col * dest->nChannels) + RED] =  (unsigned char) (r / cells);
	dest->imageData[(ren * step) + (col * dest->nChannels) + GREEN] = (unsigned char) (g / cells);
	dest->imageData[(ren * step) + (col * dest->nChannels) + BLUE] = (unsigned char) (b / cells);
}
	
void blur(IplImage *src, IplImage *dest) {
	int index, size;
    int ren, col;
    
    size = src->width * src->height;
    #pragma omp parallel for shared(size, src, dest) private(ren, col)
    for (index = 0; index < size; index++) {
    	ren = index / src->width;
    	col = index % src->width;
    	blur_pixel(src, dest, ren, col);
    }
}

int main(int argc, char* argv[]) {
	int i;
	double acum; 	
	
	if (argc != 2) {
		printf("usage: %s source_file\n", argv[0]);
		return -1;
	}
	
	IplImage *src = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
	IplImage *dest = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 3);
	if (!src) {
		printf("Could not load image file: %s\n", argv[1]);
		return -1;
	}
	
	acum = 0;
	for (i = 0; i < N; i++) {
		start_timer();
		blur(src, dest);
		acum += stop_timer();
	}
	
	printf("avg time = %.5lf ms\n", (acum / N));
	
	cvShowImage("Lenna (Original)", src);
	cvShowImage("Lenna (Blur)", dest);
	cvWaitKey(0);
	cvDestroyWindow("Lenna (Original)");
	cvDestroyWindow("Lenna (Blur)");

	return 0;
}
