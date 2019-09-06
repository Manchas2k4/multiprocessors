/* This code will generate a fractal image. Uses OpenCV, to compile:
   g++ example7.cpp `pkg-config --cflags --libs opencv`  */
   
#include <iostream>
#include <opencv/highgui.h>
#include "utils/cppheader.h"

using namespace std;

const int WIDTH = 1024;
const int HEIGHT = 768;
const int SCALEX = 1.500;
const int SCALEY = 1.500;

enum color {BLUE, GREEN, RED};

class Complex {
private:
	float real, img;
	
public:
	Complex(float r, float i) : real(r), img(i) {}
	
	float magnitude2() const {
		return (real * real) + (img * img);
	}
	
	Complex operator*(const Complex &a) {
		return Complex( ((real * a.real) - (img * a.img)),
		 				((img * a.real) + (real * a.img)) );
	}
	
	Complex operator+(const Complex &a) {
		return Complex( (real + a.real),
		 				(img + a.img) );
	}
};

class JuliaSet {
private:
	IplImage *img;
	
	int juliaValue(int x, int y, int width, int height) {
		int k;
		float jx = SCALEX * (float) (width / 2 - x) / (width / 2);
		float jy = SCALEY * (float) (height / 2 - y) / (height / 2);
		Complex c(-0.8, 0.156);
		Complex a(jx, jy);
	 
		for (k = 0; k < 200; k++) {
		    a = a*a + c;
		    if (a.magnitude2() > 1000) {
		        return 0;
		    }
		}
		return 1;
	}
	
public:
	JuliaSet(IplImage *image) : img(image) {}
	
	void doMagic() {
		int index, size, step;
		int ren, col, value;
		
		size = img->width * img->height;
		step = img->widthStep / sizeof(uchar);
		for (index = 0; index < size; index++) {
			ren = index / img->width;
			col = index % img->width;
			
			value = juliaValue(col, ren, img->width, img->height);
			
			img->imageData[(ren * step) + (col * img->nChannels) + RED] = (unsigned char) (255 * (0.4 * value));
			img->imageData[(ren * step) + (col * img->nChannels) + GREEN] = (unsigned char) (255 * (0.5 * value));
			img->imageData[(ren * step) + (col * img->nChannels) + BLUE] = (unsigned char) (255 * (0.7 * value));
		}
	}
};

int main(int argc, char* argv[]) {
	Timer t;
	double ms; 	
	IplImage* img=cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 3);
	
	JuliaSet js(img);
	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();
		js.doMagic();
		ms += t.stop();
	}
	cout << "avg time = " << (ms /N) << endl;
	
	cvShowImage("CPU Julia | c(-0.8, 0.156)", img);
    cvWaitKey(0);
    cvDestroyWindow("CPU Julia | c(-0.8, 0.156)");
	
	return 0;
}
