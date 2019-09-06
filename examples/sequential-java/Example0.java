/* This code adds all the values of an array */

public class Example0 {
	private int array[];
	private long result;
	
	public Example0(int array[]) {
		this.array = array;
		this.result = 0;
	}
	
	public long getResult() {
		return result;
	}
	
	public void calculate() {
		result = 0;
		for (int i = 0; i < array.length; i++) {
			result += array[i];
		}
	}
	
	public static void main(String args[]) {
		final int SIZE = 300_000_000;
		int array[] = new int[SIZE];
		long startTime, stopTime;
		double acum = 0;
		
		Utils.fillArray(array);
		Utils.displayArray("array", array);
		
		Example0 e = new Example0(array);
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			e.calculate();
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		System.out.printf("sum = %d\n", e.getResult());
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
