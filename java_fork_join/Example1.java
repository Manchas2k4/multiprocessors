/* This code adds all the values of an array */
import java.util.concurrent.RecursiveTask;

public class Example1 extends RecursiveTask<Long> {
	private static final long MIN = 100_000;
	private int array[], start, end;
	
	public Example1(int array[], int start, int end) {
		this.array = array;
		this.start = start;
		this.end = end;
	}
	
	public Long computeDirectly() {
		long sum = 0;
		
		for (int i = start; i < end; i++) {
			sum += array[i];
		}
	}
	
	@Override
	protected Long compute() {
		if (end - start <= Example1.MIN) {
			return computeDirectly();
		} else {
			int mid = (end + start) / 2;
			Example1 lowerMid = new Example1(array, start, mid);
			lowerMid.fork();
			Example1 upperMid = new Example1(array, mid, end);
			return (upperMid.compute() + lowerMid.join());
		}
	}
}
			
