/* This code calculates an approximation of PI */

#include <iostream>
#include "utils/cppheader.h"

using namespace std;

const long NUM_RECTS = 100000000;

class CalculatingPi {
private:
	double area;

public:
	CalculatingPi() : area(0) {}

	double getPi() const {
		return area;
	}

	void calculate() {
		double mid, height, width;
		double sum = 0;

		width = 1.0 / (double) NUM_RECTS;
		for (int i = 0; i < NUM_RECTS; i++) {
			mid = (i + 0.5) * width;
			height = 4.0 / (1.0 + (mid * mid));
			sum += height;
		}
		area = width * sum;
	}
};

int main(int argc, char* argv[]) {
	CalculatingPi cp;
	Timer t;
	double ms;

	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		t.start();
		cp.calculate();
		ms += t.stop();
	}
	cout << "PI = " << cp.getPi() << endl;
	cout << "avg time = " << (ms / N) << " ms" << endl;

	return 0;
}
