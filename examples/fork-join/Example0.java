/* This code adds all the values of an array */
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class Example0 extends RecursiveTask<Long> {
	private static final int SIZE = 300_000_000;
	private static final int MIN = 1_000_000;
	private int array[], start, end;
	
	public Example0(int start, int end, int array[]) {
		this.start = start;
		this.end = end;
		this.array = array;
	}
		
	protected Long computeDirectly() {
		long result = 0;
		for (int i = start; i < end; i++) {
			result += array[i];
		}
		return result;
	}
	
	@Override 
	protected Long compute() {
		if ( (end - start) <= MIN ) {
			return computeDirectly();
		} else {
			int mid = start + ( (end - start) / 2 );
			Example0 lowerMid = new Example0(start, mid, array);
			lowerMid.fork();
			Example0 upperMid = new Example0(mid, end, array);
			return upperMid.compute() + lowerMid.join();
		}
	}
	
	public static void main(String args[]) {
		ForkJoinPool pool;
		int array[] = new int[SIZE];
		long startTime, stopTime, result = 0;
		double acum = 0;
		
		Utils.fillArray(array);
		Utils.displayArray("array", array);
		
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(Utils.MAXTHREADS);
			result = pool.invoke(new Example0(0, array.length, array)); 
			
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		System.out.printf("sum = %d\n", result);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
