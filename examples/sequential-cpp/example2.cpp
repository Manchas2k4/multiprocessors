/* This code calculates an arctan approximation for |x| < 1 */
#include <iostream>
#include <cmath>
#include "utils/cppheader.h"

const int LIMIT = 1000000000;

using namespace std;

class Atan {
private:
	double myX, result;
	
public:
	Atan(double x) : myX(x), result(0) {}
	
	double getResult() const {
		return result;
	}
	
	void calculate() {
		double one, n;
		
		result = 0;
		for (int j = 0; j < LIMIT; j++) {
			one = (j % 2 == 0)? 1.0 : -1.0;
			n = (2 * j ) + 1;
			result += ( (one / n) * pow(myX, n) );
		}
	}
};

int main(int argc, char* argv[]) {
	double ms;
	Timer t;
	
	ms = 0;
	Atan obj(0.99);
	cout << "Starting..." << endl;
	for (int i = 0; i < N; i++) {
		t.start();
		obj.calculate();
		ms += t.stop();
	}
	cout << "arctan(0.99)->(0.78) = " << obj.getResult() << endl;
	cout << "avg time = " << (ms / N) << " ms" << endl;
	
	return 0;
}
