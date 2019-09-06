/* This code adds all the values of an array */

public class Example0 extends Thread {
	private int array[], start, end;
	private long result;
	
	public Example0(int array[], int start, int end) {
		super();
		this.array = array;
		this.start = start;
		this.end = end;
		this.result = 0;
	}
	
	public long getResult() {
		return result;
	}
	
	public void run() {
		result = 0;
		for (int i = start; i < end; i++) {
			result += array[i];
		}
	}
	
	public static void main(String args[]) {
		final int SIZE = 300_000_000;
		Example0 threads[];
		int block;
		long startTime, stopTime, total = 0;
		double acum = 0;
		
		int array[] = new int[SIZE];
		Utils.fillArray(array);
		Utils.displayArray("array", array);
		
		block = SIZE / Utils.MAXTHREADS;
		threads = new Example0[Utils.MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example0(array, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example0(array, (i * block), SIZE);
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
			
			if (j == Utils.N) {
				total = 0;
				System.out.printf("sum1 = %d\n", total);
				for (int i = 0; i < threads.length; i++) {
					total += threads[i].getResult();
				}
				System.out.printf("sum2 = %d\n", total);
			}
		}
		System.out.printf("sum = %d\n", total);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
