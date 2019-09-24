/* This code adds all the values of an array */
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example4 extends RecursiveAction {
	private static final int SIZE = 100_000_000;
	private static final int MIN = 100_000;
	private int a[], b[], c[], start, end;
	
	public Example4(int start, int end, int c[], int a[], int b[]) {
		this.start= start;
		this.end = end;
		this.a = a;
		this.b = b;
		this.c = c;
	}
	
	protected void computeDirectly() {
		for (int i = start; i < end; i++) {
			c[i] = a[i] + b[i];
		}
	}
	
	@Override 
	protected void compute() {
		if ( (end - start) <= MIN ) {
			computeDirectly();
		} else {
			int mid = start + ( (end - start) / 2 );
			invokeAll(
				new Example4(start, mid, c, a, b),
				new Example4(mid, end, c, a, b)
			);
		}
	}
	
	
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
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(Utils.MAXTHREADS);
			pool.invoke(new Example4(0, SIZE, c, a, b));
			
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
