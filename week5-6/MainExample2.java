/* This code adds all the values of an array */
import java.util.concurrent.ForkJoinPool;

public class MainExample2 {
	private static int SIZE = 100_000_000;
	private static final int MAXTHREADS = Runtime.getRuntime().availableProcessors();
	
	public static void main(String args[]) {
		ForkJoinPool pool;
		long startTime, stopTime;
		double acum = 0;
		
		int a[] = new int[SIZE];
		Utils.fillArray(a);
		Utils.displayArray("a", a);
		
		int b[] = new int[SIZE];
		Utils.fillArray(b);
		Utils.displayArray("b", b);
		
		int c[] = new int[SIZE];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(MAXTHREADS);
			pool.invoke(new Example2(c, b, a, 0, SIZE));
			
			stopTime = System.currentTimeMillis();
			acum +=  (stopTime - startTime);
		}
		
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
