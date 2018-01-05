/* This code will generate a fractal image. Uses OpenCV, to compile:
   nvcc example5.c `pkg-config --cflags --libs opencv`  */
#include <stdio.h>
#include <stdlib.h>
#include <opencv/highgui.h>
#include "utils/cheader.h"

#define BLUR_WINDOW 15

typedef enum color {BLUE, GREEN, RED} Color;

__global__ void blur(const unsigned char *src, unsigned char *dest, int width, int height, int step, int nChannels) {
	int side_pixels, i, j, cells;
	int tmp_ren, tmp_col;
	float r, g, b;
	
	int ren = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	
	side_pixels = (BLUR_WINDOW - 1) / 2;
	cells = (BLUR_WINDOW * BLUR_WINDOW);
	r = 0; g = 0; b = 0;
	for (i = -side_pixels; i <= side_pixels; i++) {
		for (j = -side_pixels; j <= side_pixels; j++) {
			tmp_ren = MIN( MAX(ren + i, 0), height - 1 );
			tmp_col = MIN( MAX(col + j, 0), width - 1);
			
			r += (float) src[(tmp_ren * step) + (tmp_col * nChannels) + RED];
			g += (float) src[(tmp_ren * step) + (tmp_col * nChannels) + GREEN];
			b += (float) src[(tmp_ren * step) + (tmp_col * nChannels) + BLUE];
		}
	}
	
	dest[(ren * step) + (col * nChannels) + RED] =  (unsigned char) (r / cells);
	dest[(ren * step) + (col * nChannels) + GREEN] = (unsigned char) (g / cells);
	dest[(ren * step) + (col * nChannels) + BLUE] = (unsigned char) (b / cells);
}
	
int main(int argc, char* argv[]) {
	int i;
	double acum; 	
	unsigned char *d_src, *d_dest; 
	
	if (argc != 2) {
		printf("usage: %s source_file\n", argv[0]);
		return -1;
	}
	
	IplImage *src_img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
	IplImage *dest_img = cvCreateImage(cvSize(src_img->width, src_img->height), IPL_DEPTH_8U, 3);
	if (!src_img) {
		printf("Could not load image file: %s\n", argv[1]);
		return -1;
	}
	
	cudaMalloc((void**) &d_src, sizeof(unsigned char) * src_img->width * src_img->height * src_img->nChannels);
	cudaMalloc((void**) &d_dest, sizeof(unsigned char) * src_img->width * src_img->height * src_img->nChannels);
	
	cudaMemcpy(d_src, src_img->imageData, sizeof(unsigned char) * src_img->width * src_img->height * src_img->nChannels, cudaMemcpyHostToDevice);
	
	int step = src_img->widthStep / sizeof(uchar);
	dim3 blockSize(16,16);
	dim3 gridSize(src_img->width/blockSize.x+1,src_img->height/blockSize.y+1);
	
	acum = 0;
	for (i = 0; i < N; i++) {
		start_timer();
		blur<<<gridSize, blockSize>>>(d_src, d_dest, src_img->width, src_img->height, step, src_img->nChannels);
		acum += stop_timer();
	}
	
	cudaMemcpy(dest_img->imageData, d_dest, sizeof(unsigned char) * src_img->width * src_img->height * src_img->nChannels, cudaMemcpyDeviceToHost);
	
	printf("avg time = %.5lf ms\n", (acum / N));
	
	cvShowImage("Lenna (Original)", src_img);
	cvShowImage("Lenna (Blur)", dest_img);
	cvWaitKey(0);
	cvDestroyWindow("Lenna (Original)");
	cvDestroyWindow("Lenna (Blur)");
	
	return 0;
}
