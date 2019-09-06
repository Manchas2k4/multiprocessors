/* This code calculates the factorial of a integer number. */

import java.math.BigInteger;

public class Example9 {
	private BigInteger result;
	private int n;
	
	public Example9(int val) {
		n = val;
	}
	
	public BigInteger getResult() {
		return result;
	}
	
	public void calculate() {
		result = BigInteger.valueOf(1);
		for (int i = 1; i <= n; i++) {
			result = result.multiply(BigInteger.valueOf(i));
		}
	}
	
	public static void main(String args[]) {
		final int NUM = 100_000;
		long startTime, stopTime;
		double acum = 0;
		
		Example9 e = new Example9(NUM);
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			e.calculate();
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		System.out.println("factorial(" + NUM + ") = " + e.getResult());
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
