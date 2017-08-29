/* This code adds all the values of an array */

public class MainExample1 {
	private static int SIZE = 300_000_000;
	private static final int MAXTHREADS = 4;
	
	public static void main(String args[]) {
		Example1 threads[];
		int block;
		long startTime, stopTime, total = 0;
		double acum = 0;
		
		int array[] = new int[SIZE];
		Utils.fillArray(array);
		Utils.displayArray("array", array);
		
		block = SIZE / MAXTHREADS;
		threads = new Example1[MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example1(array, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example1(array, (i * block), SIZE);
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
				total = 0;
				System.out.printf("sum1 = %d\n", total);
				for (int i = 0; i < threads.length; i++) {
					total += threads[i].getResult();
				}
				System.out.printf("sum2 = %d\n", total);
			}
		}
		System.out.printf("sum = %d\n", total);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}