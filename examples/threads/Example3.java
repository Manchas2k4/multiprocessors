/* This code calculate the deviation of a integer numbers */
class Average extends Thread{
	private int array[], start, end;
	private double result;
	
	public Average(int array[], int start, int end) {
		this.array = array;
		this.start = start;
		this.end = end;
		this.result = 0;
	}
	
	public double getResult() {
		return result;
	}
	
	public void run() {
		result = 0;
		for (int i = start; i < end; i++) {
			result += array[i];
		}
	}
}

class Deviation extends Thread{
	private int array[], start, end;
	private double result, avg;
	
	public Deviation(int array[], double avg, int start, int end) {
		this.array = array;
		this.avg = avg;
		this.start = start;
		this.end = end;
		this.result = 0;
	}
	
	public double getResult() {
		return result;
	}
	
	public void run() {
		result = 0;
		for (int i = start; i < end; i++) {
			result += (array[i] - avg) * (array[i] - avg);
		}
	}
}

public class Example3 {
	public static void main(String args[]) {
		final int SIZE = 300_000_000;
		Average avgs[];
		Deviation devs[];
		int block, i;
		long startTime, stopTime;
		double avg = 0, dev = 0, acum = 0;
		
		int array[] = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("array", array);
		
		block = SIZE / Utils.MAXTHREADS;
		avgs = new Average[Utils.MAXTHREADS];
		devs = new Deviation[Utils.MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (i = 0; i < avgs.length; i++) {
				if (i != avgs.length - 1) {
					avgs[i] = new Average(array, (i * block), ((i + 1) * block));
				} else {
					avgs[i] = new Average(array, (i * block), ((i + 1) * block));
				}
			}
			
			startTime = System.currentTimeMillis();
			for (i = 0; i < avgs.length; i++) {
				avgs[i].start();
			}
			for (i = 0; i < avgs.length; i++) {
				try {
					avgs[i].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			avg = 0;
			for (i = 0; i < avgs.length; i++) {
				avg += avgs[i].getResult();
			}
			avg = avg / array.length;
			
			for (i = 0; i < devs.length; i++) {
				if (i != devs.length - 1) {
					devs[i] = new Deviation(array, avg, (i * block), ((i + 1) * block));
				} else {
					devs[i] = new Deviation(array, avg, (i * block), ((i + 1) * block));
				}
			}
			
			for (i = 0; i < devs.length; i++) {
				devs[i].start();
			}
			for (i = 0; i < devs.length; i++) {
				try {
					devs[i].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			stopTime = System.currentTimeMillis();
			acum +=  (stopTime - startTime);
			
			if (j == Utils.N) {
				dev = 0;
				for (i = 0; i < devs.length; i++) {
					dev += devs[i].getResult();
				}
				dev = Math.sqrt(dev / array.length);
			}
		}
		System.out.printf("S = %f\n", dev);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
