/* This code will generate a fractal image. Uses OpenCV, to compile:
   nvcc example5.cu `pkg-config --cflags --libs opencv`  */
#include <stdio.h>
#include <stdlib.h>
#include <opencv/highgui.h>
#include "utils/cheader.h"

#define BLUR_WINDOW 15

typedef enum color {BLUE, GREEN, RED} Color;

__global__ void blur(unsigned char *src, unsigned char *dest, int width, int heigth, int blur_window, int step, int channels) { 
	int i, j, side_pixels, cells;
	int ren, col, tmp_ren, tmp_col;
	float r, g, b;
	
	ren = blockIdx.x;
	col = threadIdx.x;
	side_pixels = (blur_window - 1) / 2;
	cells = (blur_window * blur_window);
	r = 0; g = 0; b = 0;
	for (i = -side_pixels; i <= side_pixels; i++) {
		for (j = -side_pixels; j <= side_pixels; j++) {
			tmp_ren = MIN( MAX(ren + i, 0), heigth - 1 );
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
	int i, step, size;
	double acum; 
	unsigned char *dev_src, *dev_dest;
		
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
	
	size = src->width * src->height * src->nChannels * sizeof(uchar);
	cudaMalloc((void**) &dev_src, size);
	cudaMalloc((void**) &dev_dest, size);
	
	cudaMemcpy(dev_src, src->imageData, size, cudaMemcpyHostToDevice);
	
	acum = 0;
	step = src->widthStep / sizeof(uchar);
	printf("Starting...\n");
	for (i = 0; i < N; i++) {
		start_timer();
		blur<<<src->height, src->width>>>(dev_src, dev_dest, src->width, src->height, BLUR_WINDOW, step, src->nChannels);
		acum += stop_timer();
	}
	
	cudaMemcpy(dest->imageData, dev_dest, size, cudaMemcpyDeviceToHost);
	
	cudaFree(dev_dest);
	cudaFree(dev_src);
	
	printf("avg time = %.5lf ms\n", (acum / N));
	
	cvShowImage("Lenna (Original)", src);
	cvShowImage("Lenna (Blur)", dest);
	cvWaitKey(0);
	cvDestroyWindow("Lenna (Original)");
	cvDestroyWindow("Lenna (Blur)");

	return 0;
}
