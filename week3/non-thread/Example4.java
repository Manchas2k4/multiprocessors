/* This code calculates the factorial of a integer number. */

import java.math.BigInteger;

public class Example4 {
	private BigInteger result;
	private int n;
	
	public Example4(int val) {
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
}
