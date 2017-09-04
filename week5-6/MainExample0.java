/* This code adds all the values of an array */
import java.util.concurrent.ForkJoinPool;

public class MainExample0 {
	private static int NUM_RECTS = 100_000_000;
	private static final int MAXTHREADS = Runtime.getRuntime().availableProcessors();
	
	public static void main(String args[]) {
		ForkJoinPool pool; 
		long startTime, stopTime; 
		double total = 0, acum = 0, width;
		
		width = 1.0 / (double) NUM_RECTS;
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(MAXTHREADS);
			total = pool.invoke(new Example0(0, NUM_RECTS, width));
			
			stopTime = System.currentTimeMillis();
			acum +=  (stopTime - startTime);
		}
		System.out.printf("PI = %.5f\n", total);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
