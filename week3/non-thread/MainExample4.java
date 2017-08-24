/* This code calculates the factorial of a integer number. */

import java.math.BigInteger;

public class MainExample4 {
	private static final int NUM = 100_000;
	
	public static void main(String args[]) {
		long startTime, stopTime;
		double acum = 0;
		
		Example4 e = new Example4(NUM);
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
