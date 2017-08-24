/* This code adds all the values of an array */

public class MainExample1 {
	private static int SIZE = 300_000_000;
	
	public static void main(String args[]) {
		int array[] = new int[SIZE];
		long startTime, stopTime;
		double acum = 0;
		
		Utils.fillArray(array);
		Utils.displayArray("array", array);
		
		Example1 e = new Example1(array);
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