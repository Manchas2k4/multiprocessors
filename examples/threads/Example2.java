/* This code adds all the values of an array */

public class Example2 extends Thread {
	private double result, x;
	private int start, end;
	
	public Example2(double x, int start, int end) {
		this.x = x;
		this.start = start;
		this.end = end;
		this.result = 0;
	}
	
	public double getResult() {
		return result;
	}
	
	public void run() {
		double one, n;
		
		result = 0;
		for (int j = start; j < end; j++) {
			one = (j % 2 == 0)? 1.0 : -1.0;
			n = (2 * j ) + 1;
			result += ( (one / n) * Math.pow(x, n) );
		}
	}
	
	public static void main(String args[]) {
	 	final int LIMIT = 100_000_000;
		Example2 threads[];
		int block;
		long startTime, stopTime;
		double width, result = 0, acum = 0;
		
		block = LIMIT / Utils.MAXTHREADS;
		threads = new Example2[Utils.MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example2(0.99, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example2(0.99, (i * block), LIMIT);
				}
			}
			
			startTime = System.currentTimeMillis();
			for (int i = 0; i < threads.length; i++) {
				threads[i].start();
			}
			for (int i = 0; i < threads.length; i++) {
				try {
					threads[i].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			stopTime = System.currentTimeMillis();
			acum +=  (stopTime - startTime);
			
			if (j == Utils.N) {
				result = 0;
				for (int i = 0; i < threads.length; i++) {
					result += threads[i].getResult();
				}
			}
		}
		System.out.printf("arctan(0.99)->(0.78) = %f\n", result);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
