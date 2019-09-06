/* This code adds all the values of an array */

public class Example4 extends Thread {
	private int a[], b[], c[], start, end;
	
	public Example4(int c[], int a[], int b[], int start, int end) {
		this.a = a;
		this.b = b;
		this.c = c;
		this.start = start;
		this.end = end;
	}
	
	public void run() {
		for (int i = start; i < end; i++) {
			c[i] = a[i] + b[i];
		}
	}
	
	public static void main(String args[]) {
		final int SIZE = 100_000_000;
		Example4 threads[];
		int block;
		long startTime, stopTime;
		double acum = 0;
		
		int a[] = new int[SIZE];
		Utils.fillArray(a);
		Utils.displayArray("a", a);
		
		int b[] = new int[SIZE];
		Utils.fillArray(b);
		Utils.displayArray("b", b);
		
		int c[] = new int[SIZE];
		
		block = SIZE / Utils.MAXTHREADS;
		threads = new Example4[Utils.MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example4(c, b, a, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example4(c, b, a, (i * block), SIZE);
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
		}
		
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
