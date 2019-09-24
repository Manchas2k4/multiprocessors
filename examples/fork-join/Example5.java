/* This code implements the known sort algorithm "Counting sort" */
import java.util.Arrays;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example5 extends RecursiveAction {
	private static final int SIZE = 10_000;
	private static final int MIN = 5_000;
	private int array[], temp[], start, end;
	
	public Example5(int start, int end, 
					int array[], int temp[]) {
		this.array = array;
		this.temp = temp;
		this.start = start;	
		this.end = end;
	}
	
	protected void computeDirectly()  {
		int i, j, count;
		
		for (i = start; i < end; i++) {
			count = 0;
			for (j = 0; j < array.length; j++) {
				if (array[j] < array[i]) {
					count++;
				} else if (array[i] == array[j] && j < i) {
					count++;
				}
			}
			temp[count] = array[i];
		}
	}

	@Override 
	protected void compute() {
		if ( (end - start) <= MIN ) {
			computeDirectly();
		} else {
			int mid = (end - start)/2;
			invokeAll(
				new Example5(start, mid, array, temp),
				new Example5(mid, end, array, temp)
			);
		}
	}
	
	public static void main(String args[]) {
		
		long startTime, stopTime;
		ForkJoinPool pool;
		double acum = 0;
		
		int array[] = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("before", array);

		int temp[] = new int[SIZE];
		
		acum = 0;
		for (int i = 1; i <= Utils.N; i++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			pool.invoke(
				new Example5(0, array.length, 
					array, temp)
			);

			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		array = Arrays.copyOf(temp, temp.length);

		Utils.displayArray("after", array);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
