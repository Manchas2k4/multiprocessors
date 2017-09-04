/* This code adds all the values of an array */
import java.util.concurrent.ForkJoinPool;

public class MainExample1 {
	private static int SIZE = 300_000_000;
	private static final int MAXTHREADS = Runtime.getRuntime().availableProcessors();
	
	public static void main(String args[]) {
		ForkJoinPool pool; 
		long startTime, stopTime, total = 0;
		double acum = 0;
		
		int array[] = new int[SIZE];
		Utils.fillArray(array);
		Utils.displayArray("array", array);
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(MAXTHREADS);
			total = pool.invoke(new Example1(array, 0, array.length));
			
			stopTime = System.currentTimeMillis();
			acum +=  (stopTime - startTime);
		}
		System.out.printf("sum = %d\n", total);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
