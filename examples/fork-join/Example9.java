/* This code calculates the factorial of a integer number. */

import java.math.BigInteger;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class Example9 extends RecursiveTask<BigInteger> {
	private static final int MIN = 10_000;
	private int start, end;
	
	public Example9(int start, int end) {
		this.start = start;
		this.end = end;
	}
	
	protected BigInteger computeDirectly() {
		BigInteger result = BigInteger.valueOf(1);
		for (int i = start; i <= end; i++) {
			result = result.multiply(BigInteger.valueOf(i));
		}
		return result;
	}

	@Override
	protected BigInteger compute() {
		if ( (end - start) <= MIN ) {
			return computeDirectly();
		} else {
			int mid = (end + start)/2;
			Example9 lowerMid = new Example9(start, mid);
			lowerMid.fork();
			Example9 upperMid = new Example9(mid + 1, end);
			return upperMid.compute().multiply(lowerMid.join());
		}
	}
	
	public static void main(String args[]) {
		final int NUM = 100_000;
		ForkJoinPool pool;
		long startTime, stopTime;
		double acum = 0;
		BigInteger aux = BigInteger.ONE;
		
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(Utils.MAXTHREADS);
			aux = pool.invoke(new Example9(1, NUM));

			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		System.out.println("factorial(" + NUM + ") = " + aux);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
