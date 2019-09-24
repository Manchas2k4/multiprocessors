/* This code implements the known sort algorithm "Counting sort" */
import java.util.Arrays;

public class Example5 extends Thread {
	private int array[], temp[], start, end;
	
	public Example5(int array[], int start, int end) {
		super();
		this.array = array;
		this.start = start;
		this.end = end;
		this.temp = new int[array.length];
	}
	
	public void copyArray() {
		for (int i = 0; i < temp.length; i++) {
			if (temp[i] != 0) {
				array[i] = temp[i];
			}
		}
	}
	
	public void run() {
		
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
	
	public static void main(String args[]) {
		final int SIZE = 10_000;
		
		Example5 threads[];
		int block;
		long startTime, stopTime;
		double acum = 0;
		
		int array[] = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("before", array);
		
		block = SIZE / Utils.MAXTHREADS;
		threads = new Example5[Utils.MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example5(array, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example5(array, (i * block), SIZE);
				}
			}
			
			startTime = System.currentTimeMillis();
			for (int i = 0; i < threads.length; i++) {
				threads[i].start();
			}
			for (int i = 0; i < threads.length; i++) {
				try {
					threads[i].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			stopTime = System.currentTimeMillis();
			acum +=  (stopTime - startTime);
		}
		
		Arrays.fill(array, 0);
		for (int i = 0; i < threads.length; i++) {
			threads[i].copyArray();
		}
		
		Utils.displayArray("after", array);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
