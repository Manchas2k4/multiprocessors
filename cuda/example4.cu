/* This code will generate a fractal image. Uses OpenCV, to compile:
  nvcc example4-4.cu -lGL -lGLU -lglut */
#include <stdio.h>
#include <stdlib.h>
#include <opencv/highgui.h>
#include "utils/cheader.h"

#define WIDTH 	1024
#define HEIGHT 	768
#define SCALEX	1.500
#define SCALEY 	1.500

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

enum color {RED, GREEN, BLUE, ALPHA};

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

__global__ void julia_set(unsigned char *img, int width, int height, int step, int channels) {
    int col = threadIdx.x;
    int ren = blockIdx.x;

    // now calculate the value at that position
    int value = julia_value(col, ren, width, height);
    img[(ren * step) + (col * channels) + RED] = (unsigned char) (255 * (0.4 * value));
    img[(ren * step) + (col * channels) + GREEN] = (unsigned char) (255 * (0.5 * value));
    img[(ren * step) + (col * channels) + BLUE] = (unsigned char) (255 * (0.7 * value));
    img[(ren * step) + (col * channels) + ALPHA] = 255;
}

struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main(int argc, char* argv[]) {
	IplImage* img=cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 3);
	unsigned char *dev_img;
    double ms;
    int size, step;

	size = img->width * img->height * img->nChannels * sizeof(uchar);
	cudaMalloc((void**) &dev_img, size);
	
	ms = 0;
	step = img->widthStep / sizeof(uchar);
	printf("Starting...\n");
	for (int i = 0; i < N; i++) {
		start_timer();
		julia_set<<<img->height, img->width>>>(dev_img, img->width, img->height, step, img->nChannels);
		ms += stop_timer();
	}

	cudaMemcpy(img->imageData, dev_img, size, cudaMemcpyDeviceToHost);
	
	cudaFree(dev_img);
	
	printf("avg time = %.5lf ms\n",  ms/N);
	
	cvShowImage("GPU Julia | c(-0.8, 0.156)", img);
    cvWaitKey(0);
    cvDestroyWindow("GPU Julia | c(-0.8, 0.156)");
    
	return 0;
}
