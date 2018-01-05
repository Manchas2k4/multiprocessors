/* This code calculates the factorial of a integer number. */
import java.math.BigInteger;
import java.util.concurrent.ForkJoinPool;

public class MainExample6 {
	private static final int NUM = 100_000;
	private static final int MAXTHREADS = Runtime.getRuntime().availableProcessors();
	
	public static void main(String args[]) {
		ForkJoinPool pool; 
		long startTime, stopTime;
		double acum = 0;
		BigInteger result = BigInteger.valueOf(1);
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(MAXTHREADS);
			result = pool.invoke(new Example6(1, NUM));
			
			stopTime = System.currentTimeMillis();
			acum +=  (stopTime - startTime);
		}
		
		System.out.println("factorial(" + NUM + ") = " + result);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
