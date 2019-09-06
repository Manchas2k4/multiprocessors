/* This code calculates the factorial of a integer number. */

import java.math.BigInteger;

public class Example9 extends Thread {
	private BigInteger result;
	private int start, end;
	
	public Example9(int start, int end) {
		super();
		this.start = start;
		this.end = end;
		this.result = BigInteger.valueOf(1);
	}
	
	public BigInteger getResult() {
		return result;
	}
	
	public void run() {
		result = BigInteger.valueOf(1);
		for (int i = start; i <= end; i++) {
			result = result.multiply(BigInteger.valueOf(i));
		}
	}
	
	public static void main(String args[]) {
		final int NUM = 100_000;
		
		Example9 threads[];
		int block;
		long startTime, stopTime;
		double acum = 0;
		BigInteger result = BigInteger.valueOf(1);
		
		block = NUM / Utils.MAXTHREADS;
		threads = new Example9[Utils.MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example9((i + 1) * block, (i + 2) * block);
				} else {
					threads[i] = new Example9((i + 1) * block, NUM);
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
