/* This code adds all the values of an array */

public class MainExample2 {
	private static int SIZE = 100_000_000;
	private static final int MAXTHREADS = 4;
	
	public static void main(String args[]) {
		Example2 threads[];
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
		
		block = SIZE / MAXTHREADS;
		threads = new Example2[MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example2(c, b, a, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example2(c, b, a, (i * block), SIZE);
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
			
