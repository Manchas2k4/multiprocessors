/* This code adds all the values of an array */
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class Example1 extends RecursiveTask<Double> {
	private static final int NUM_RECTS = 1_000_000_000;
	private static final int MIN = 1_000_000;
	private int start, end;
	private double width;
	
	public Example1(int start, int end, double width) {
		this.start = start;
		this.end = end;
		this.width = width;
	}
	
	protected Double computeDirectly() {
		double sum, mid, height;
		
		sum = 0;
		for (int j = start; j < end; j++) {
			mid = (j + 0.5) * width;
			height = 4.0 / (1.0 + (mid * mid));
			sum += height;
		}
		return sum;
	}
	
	@Override 
	protected Double compute() {
		if ( (end - start) <= MIN ) {
			return computeDirectly();
		} else {
			int mid = start + ( (end - start) / 2 );
			Example1 lowerMid = new Example1(start, mid, width);
			lowerMid.fork();
			Example1 upperMid = new Example1(mid, end, width);
			return upperMid.compute() + lowerMid.join();
		}
	}
	
	public static void main(String args[]) {
		ForkJoinPool pool;
		long startTime, stopTime;
		double acum = 0, result = 0, width;
		
		width = 1.0 / (double) NUM_RECTS;
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(Utils.MAXTHREADS);
			result = pool.invoke(new Example1(0, NUM_RECTS, width)); 
			
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		System.out.printf("PI = %f\n", (result * width));
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
