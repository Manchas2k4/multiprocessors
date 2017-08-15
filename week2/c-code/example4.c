#include <stdio.h>
#include <stdlib.h>
#include <opencv/highgui.h>
#include "utils/cheader.h"

#define WIDTH 	1024
#define HEIGHT 	768
#define SCALEX	1.500
#define SCALEY	1.500

enum color {BLUE, GREEN, RED};

typedef struct complex {
	float real, img;
} Complex;

float magnitude2(const Complex *a) {
	return (a->real * a->real) + (a->img * a->img);
}
