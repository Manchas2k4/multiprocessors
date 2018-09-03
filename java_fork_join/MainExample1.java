/* This code adds all the values of an array */
import java.util.concurrent.ForkJoinPool;

public class MainExample1 {
	private static int SIZE = 300_000_000;
	private static final int MAXTHREADS = Runtime.getRuntime().availableProcessors();
	
	public static void main(String args[]) {
		ForkJoinPool pool;
		long startTime, stopTime, total;
		double ms;
		
		int array[] = new int[SIZE];
		Utils.fillArray(array);
		Utils.displayArray("array:", array);
		
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(MAXTHREADS);
			total = pool.invoke(new Example1(array, 0, array.length));
			
			stopTime = System.currentTimeMillis();
		}	
	}
}
