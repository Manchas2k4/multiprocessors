/* This code adds all the values of an array */

public class Example1 extends Thread {
	private double result, width;
	private int start, end;
	
	public Example1(double width, int start, int end) {
		this.width = width;
		this.start = start;
		this.end = end;
		this.result = 0;
	}
	
	public double getResult() {
		return result;
	}
	
	public void run() {
		double sum, mid, height;
		
		result = 0;
		for (int j = start; j < end; j++) {
			mid = (j + 0.5) * width;
			height = 4.0 / (1.0 + (mid * mid));
			result += height;
		}
	}
	
	public static void main(String args[]) {
	 	final int NUM_RECTS = 1_000_000_000;
		Example1 threads[];
		int block;
		long startTime, stopTime;
		double width, area = 0, acum = 0;
		
		block = NUM_RECTS / Utils.MAXTHREADS;
		threads = new Example1[Utils.MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			width = 1.0 / (double) NUM_RECTS;
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example1(width, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example1(width, (i * block), NUM_RECTS);
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
				area = 0;
				for (int i = 0; i < threads.length; i++) {
					area += threads[i].getResult();
				}
				area = area * width;
			}
		}
		System.out.printf("PI = %f\n", area);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
