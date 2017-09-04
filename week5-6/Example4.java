/* This code calculates the factorial of a integer number. */

import java.math.BigInteger;
import java.util.concurrent.RecursiveTask;

public class Example4 extends RecursiveTask<BigInteger> {
	private static final long MIN = 10_000;
	private int start, end;
	
	public Example4(int start, int end) {
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
		if ( (this.end - this.start) <= Example4.MIN ) {
			return this.computeDirectly();
			
		} else {
			int mid = (end + start) / 2;
			Example4 lowerMid = new Example4(start, mid);
			lowerMid.fork();
			Example4 upperMid = new Example4(mid + 1, end);
			return upperMid.compute().multiply(lowerMid.join());
		}
	}
}
