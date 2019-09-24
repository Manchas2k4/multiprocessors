/* This code calculates de atan of x, x < 1 */
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class Example2 extends RecursiveTask<Double> {
	private static final int LIMIT = 100_000_000;
	private static final int MIN = 100_000;
	private double x;
	private int start, end;
	
	public Example2(int start, int end, double x) {
		this.start = start;
		this.end = end;
		this.x = x;
	}
	
	protected Double computeDirectly() {
		double one, n, result;
		
		result = 0;
		for (int j = start; j < end; j++) {
			one = (j % 2 == 0)? 1.0 : -1.0;
			n = (2 * j ) + 1;
			result += ( (one / n) * Math.pow(x, n) );
		}
		return result;
	}
	
	@Override 
	protected Double compute() {
		if ( (end - start) <= MIN ) {
			return computeDirectly();
		} else {
			int mid = start + ( (end - start) / 2 );
			Example2 lowerMid = new Example2(start, mid, x);
			lowerMid.fork();
			Example2 upperMid = new Example2(mid, end, x);
			return upperMid.compute() + lowerMid.join();
		}
	}
	
	public static void main(String args[]) {
		ForkJoinPool pool;
		long startTime, stopTime;
		double acum = 0;
		double result = 0;
		
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(Utils.MAXTHREADS);
			result = pool.invoke(new Example2(0, LIMIT, 0.99)); 
			
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		System.out.printf("arctan(0.99)->(0.78) = %f\n", result);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
