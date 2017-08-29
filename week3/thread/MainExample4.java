/* This code calculates the factorial of a integer number. */

import java.math.BigInteger;

public class MainExample4 {
	private static final int NUM = 100_000;
	private static final int MAXTHREADS = 4;
	
	public static void main(String args[]) {
		Example4 threads[];
		int block;
		long startTime, stopTime;
		double acum = 0;
		BigInteger result = BigInteger.valueOf(1);
		
		block = NUM / MAXTHREADS;
		threads = new Example4[MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example4((i + 1) * block, (i + 2) * block);
				} else {
					threads[i] = new Example4((i + 1) * block, NUM);
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
		
		result = BigInteger.valueOf(1);
		for (int i = 0; i < threads.length; i++) {
			result = result.multiply(threads[i].getResult());
		}
				
		System.out.println("factorial(" + NUM + ") = " + result);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
