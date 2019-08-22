/* This code will generate a fractal image. Uses OpenCV, to compile:
   gcc example4.c `pkg-config --cflags --libs opencv`  */
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils/cheader.h"
 
#define WIDTH	1024
#define HEIGHT	768
#define SCALEX	1.500
#define SCALEY	1.500
#define N       10  

typedef enum color {BLUE, GREEN, RED} Color;
 
typedef struct complex {
    float real, img;
} Complex;
 
float magnitude2(const Complex *a) {
    return (a->real * a->real) + (a->img * a->img);
}
 
void mult(Complex *result, const Complex *a, const Complex *b) {
    result->real = (a->real * b->real) - (a->img * b->img);
    result->img = (a->img * b->real) + (a->real * b->img);
}
 
void add(Complex *result, const Complex *a, const Complex *b) {
    result->real = a->real + b->real;
    result->img = a->img + b->img;
}

int julia_value(int x, int y, int width, int height) {
    int k;
    float jx = SCALEX * (float) (width / 2 - x) / (width / 2);
    float jy = SCALEY * (float) (height / 2 - y) / (height / 2);
    Complex c = {-0.8, 0.156};
    Complex a = {jx, jy};
    Complex aux;
 
    for (k = 0; k < 200; k++) {
        mult(&aux, &a, &a);
        add(&a, &aux, &c);
        if (magnitude2(&a) > 1000) {
            return 0;
        }
    }
    return 1;
}
 
void build_julia_set(IplImage* img) {
    int index, size, step;
    int ren, col, value;
    
    size = img->width * img->height;
    step = img->widthStep / sizeof(uchar);
    for (index = 0; index < size; index++) {
    	ren = index / img->width;
    	col = index % img->width;
    	
    	value = julia_value(col, ren, img->width, img->height);
    	
    	img->imageData[(ren * step) + (col * img->nChannels) + RED] = (unsigned char) (255 * (0.4 * value));
    	img->imageData[(ren * step) + (col * img->nChannels) + GREEN] = (unsigned char) (255 * (0.5 * value));
    	img->imageData[(ren * step) + (col * img->nChannels) + BLUE] = (unsigned char) (255 * (0.7 * value));
    }
}

int main(int argc, char* argv[]) {
    int i;
    double acum;    
    IplImage* img=cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 3);
 
 	acum = 0;
    for (i = 0; i < N; i++) {
        start_timer();
        build_julia_set(img);
        acum += stop_timer();
    }
     
    printf("avg time = %.5lf ms\n", (acum / N));
	
	build_julia_set(img); 
	cvShowImage("CPU Julia | c(-0.8, 0.156)", img);
    cvWaitKey(0);
    cvDestroyWindow("CPU Julia | c(-0.8, 0.156)");
    
    return 0;
}
