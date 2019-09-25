/* This code adds all the values of an array */
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

class Average extends RecursiveTask<Long> {
	private static final int MIN = 100_000;
	private int array[], start, end;
	
	public Average(int start, int end, int array[]) {
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
			Average lowerMid = new Average(start, mid, array);
			lowerMid.fork();
			Average upperMid = new Average(mid, end, array);
			return upperMid.compute() + lowerMid.join();
		}
	}
}

class Deviation extends RecursiveTask<Double> {
	private static final int MIN = 100_000;
	private int array[], start, end;
	private double average;
	
	public Deviation(int start, int end, int array[], double average) {
		this.start = start;
		this.end = end;
		this.array = array;
		this.average = average;
	}
		
	protected Double computeDirectly() {
		double result = 0;
		for (int i = start; i < end; i++) {
			result += (array[i] - average) * (array[i] - average);
		}
		return result;
	}
	
	@Override 
	protected Double compute() {
		if ( (end - start) <= MIN ) {
			return computeDirectly();
		} else {
			int mid = start + ( (end - start) / 2 );
			Deviation lowerMid = new Deviation(start, mid, array, average);
			lowerMid.fork();
			Deviation upperMid = new Deviation(mid, end, array, average);
			return upperMid.compute() + lowerMid.join();
		}
	}
}

public class Example3 {
	private static final int SIZE = 300_000_000;
	
	public static void main(String args[]) {
		int array[] = new int[SIZE];
		long startTime, stopTime;
		double acum = 0, average = 0, dev = 0;
		ForkJoinPool pool;
		
		Utils.randomArray(array);
		Utils.displayArray("array", array);
		
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(Utils.MAXTHREADS);
			average = pool.invoke(new Average(0, array.length, array)) / (double) SIZE;
			dev = pool.invoke(new Deviation(0, array.length, array, average));
			dev = Math.sqrt(dev / SIZE);
			
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		System.out.printf("S = %f\n", dev);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
