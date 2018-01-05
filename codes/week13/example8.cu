/** Compile: nvcc example8.cu -lGL -lGLU -lglut **/

#include <stdio.h>
#include <stdlib.h>
#include "utils/cpu_anim.h"

#define SIZE 	1024
#define PI		3.1415926535897932f

enum color {RED, GREEN, BLUE, ALPHA};

__global__ void kernel(unsigned char *ptr, int ticks) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	float fx = x - (SIZE / 2);
	float fy = y - (SIZE / 2);
	float d = sqrtf( (fx * fx) + (fy * fy) );
	unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                         cos(d/10.0f - ticks/7.0f) /
                                         (d/10.0f + 1.0f));
                                         
	ptr[offset*4 + RED] = grey;
	ptr[offset*4 + GREEN] = grey;
	ptr[offset*4 + BLUE] = grey;
	ptr[offset*4 + ALPHA] = 255;
}

struct DataBlock {
	unsigned char *dev_bitmap;
	CPUAnimBitmap *bitmap;
};

void generate_frame(DataBlock *d, int ticks) {
	dim3 blocks(SIZE/16, SIZE/16);
	dim3 threads(16,16);
	
	kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);
	
	cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost);
}

void cleanup( DataBlock *d ) {
    cudaFree(d->dev_bitmap); 
}

int main(int argc, char* argv[]) {
    DataBlock data;
    CPUAnimBitmap bitmap(SIZE, SIZE, &data);
    data.bitmap = &bitmap;

    cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size());

    bitmap.anim_and_exit( (void (*)(void*, int)) generate_frame,
                            (void (*)(void*)) cleanup );
                            
	return 0;
}
		
