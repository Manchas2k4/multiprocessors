/* This code adds all the values of an array */

public class Example3 {
	private int array[];
	private double result;
	
	public Example3(int array[]) {
		this.array = array;
		this.result = 0;
	}
	
	public double getResult() {
		return result;
	}
	
	public void calculate() {
		double acum, avg;
		int i;
	
		acum = 0;
		for (i = 0; i < array.length; i++) {
			acum += array[i];
		}
		avg = acum / array.length;
		
		acum = 0;
		for (i = 0; i < array.length; i++) {
			acum += (array[i] - avg) * (array[i] - avg);
		}
		
		result = Math.sqrt(acum / array.length);
	}
	
	public static void main(String args[]) {
		final int SIZE = 300_000_000;
		int array[] = new int[SIZE];
		long startTime, stopTime;
		double acum = 0;
		
		Utils.randomArray(array);
		Utils.displayArray("array", array);
		
		Example3 e = new Example3(array);
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			e.calculate();
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		System.out.printf("S = %d\n", e.getResult());
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
