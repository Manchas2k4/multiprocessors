#include <iostream>
#include <tbb/task_scheduler_init.h>
#include <tbb/task.h>
#include "timer.h"

using namespace std;
using namespace tbb;

const int MIN = 20;
const int N = 50;

long fibs[N + 1] = {0};

void iterative_fib() {
	fibs[1] = fibs[2] = 1;
	for (int i = 3; i <= N; i++) {
		fibs[i] = fibs[i - 1] + fibs[i - 2];
	}
}


long recursive_fib(long n) {
	if (n < 3) {
		return 1;
	} else {
		return recursive_fib(n - 1) + recursive_fib(n - 2);
	}
}

class FibTask : public task {
public:
	const long myN;
	long* const mySum;
	
	FibTask(long n, long *sum) : myN(n), mySum(sum) {}
	
	task* execute() {
		if (myN < MIN) {
			*mySum = recursive_fib(myN);
		} else {
			long x, y;
			
			FibTask &a = 
				*new ( allocate_child() ) FibTask(myN - 1, &x);
			FibTask &b = 
				*new ( allocate_child() ) FibTask(myN - 2, &y);
			set_ref_count(3);
			spawn(b);
			spawn_and_wait_for_all(a);
			*mySum = x + y;
		}
		return NULL;
	}
};

int main(int argc, char* argv[]) {
	long sum;
	double ms;
	Timer t;
	
	t.start();
	iterative_fib();
	ms = t.stop();
	cout << "iterative time = " << ms << " ms\n";
	cout << "sum = " << fibs[N] << "\n";
	
	t.start();
	sum = recursive_fib(N);
	ms = t.stop();
	cout << "recursive time = " << ms << " ms\n";
	cout << "sum = " << sum << "\n";
	
	t.start();
	FibTask &a = 
		*new (task::allocate_root()) FibTask(N, &sum);
	task::spawn_root_and_wait(a);
	ms = t.stop();
	cout << "TBB time = " << ms << " ms\n";
	cout << "sum = " << sum << "\n";
	return 0;
}




