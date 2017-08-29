/* This code calculates the factorial of a integer number. */

import java.math.BigInteger;

public class Example4 extends Thread {
	private BigInteger result;
	private int start, end;
	
	public Example4(int start, int end) {
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
}
