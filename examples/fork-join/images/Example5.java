/* This code implements the known sort algorithm "Counting sort" */
import java.util.Arrays;

public class Example5 {
	private int array[];
	
	public Example5(int array[]) {
		this.array = array;
	}
	
	public int[] getArray() {
		return array;
	}
	
	public void doSort(boolean copy) {
		int temp[] = new int[array.length];
		int i, j, count;
		
		for (i = 0; i < array.length; i++) {
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
		if (copy) {
			array = Arrays.copyOf(temp, temp.length);
		}
	}
	
	public static void main(String args[]) {
		final int SIZE = 10_000;
		long startTime, stopTime;
		double acum = 0;
		
		int array[] = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("before", array);
		
		Example5 e = new Example5(array);
		acum = 0;
		for (int i = 1; i <= Utils.N; i++) {
			startTime = System.currentTimeMillis();
			e.doSort((i % Utils.N == 0));
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		Utils.displayArray("after", e.getArray());
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
