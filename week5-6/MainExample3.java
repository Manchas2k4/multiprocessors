/* This code implements the known sort algorithm "Counting sort" */
import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;

public class MainExample3 {
	private static int SIZE = 10_000;
	private static final int MAXTHREADS = Runtime.getRuntime().availableProcessors();
	
	public static void main(String args[]) {
		ForkJoinPool pool;
		long startTime, stopTime;
		double acum = 0;
		
		int array[] = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("before", array);
		
		int temp[] = new int[SIZE];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(MAXTHREADS);
			pool.invoke(new Example3(array, temp, 0, array.length));
			
			stopTime = System.currentTimeMillis();
			acum +=  (stopTime - startTime);
		}
		array = Arrays.copyOf(temp, temp.length);
		Utils.displayArray("after", array);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
