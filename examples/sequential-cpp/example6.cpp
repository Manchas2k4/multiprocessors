/* This code calculates the sum of all elements in an array */
#include <iostream>
#include <cmath>
#include "utils/cppheader.h"

const int SIZE = 100000000;
const double PI = 3.14159265;
const double A = 0.2761711861;
const double B = 0.5530207692;
const double C = 0.3075097067;
const int K = 20;

using namespace std;

class Coefficient {
public:
	float real, img;
};

class FillingArrays {
private:
	int *myFunction, *myCosine, *mySine, mySize;
	
public:
	FillingArrays(int *function, int *cosine, int *sine, int size) : 
		myFunction(function), myCosine(cosine), mySine(sine), mySize(size) {}
	
	void calculate() {
		for (int j = 0; j < SIZE; j++) {
			function[j] = (((A * (double) j) + (B * (double) j)) - C);
			cosine[j] = cos((2 * j * K * PI) / N);
        	sine[j] = sin((2 * j * K * PI) / N);
		}
	}
};

class RealPart {
private:
	int *myFunction, *myCosine, mySize;
	double result;
	
public:
	RealPart(int *function, int *cosine, int size) : 
		myFunction(function), myCosine(cosine), mySize(size), result(0) {}
		
	double getResult() const {
		return result;
	}
	
	void calculate() {
		result = 0;
		for (int j = 0; j < SIZE; j++) {
			result += function[j] * cosine[j];
		}
	}
};

class ImgPart {
private:
	int *myFunction, *mySine, mySize;
	double result;
	
public:
	ImgPart(int *function, int *sine, int size) : 
		myFunction(function), mySine(sine), mySize(size), result(0) {}
		
	double getResult() const {
		return result;
	}
	
	void calculate() {
		result = 0;
		for (int j = 0; j < SIZE; j++) {
			result += function[j] * sine[j];
		}
	}
};

int main(int argc, char* argv[]) {
	double ms;
	Timer t;
	double *function, *sine, *cosine, ms;
	Coefficient coef[K];
	
	function = new double[SIZE];
	sine = new double[SIZE];
	cosine = new double [SIZE];
	
	ms = 0;
	FillingArrays fa(function, cosine, sine, SIZE);
	RealPart rp(function, cosine, SIZE);
	ImgPart im(function, sine, SIZE);
	cout << "Starting..." << endl;
	for (int i = 0; i < N; i++) {
		t.start();
		
		fa.calculate();
		for (int j = 0; j < K; j++) {
			
		}
		
		ms += t.stop();
	}
	cout << "avg = " << obj.getResult() << endl;
	cout << "avg time = " << (ms / N) << " ms" << endl;
	
	delete [] a;
	return 0;
}

