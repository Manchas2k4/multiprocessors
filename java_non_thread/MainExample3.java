/* This code implements the known sort algorithm "Counting sort" */
import java.util.Arrays;

public class MainExample3 {
	private static int SIZE = 10_000;
	
	public static void main(String args[]) {
		long startTime, stopTime;
		double acum = 0;
		
		int array[] = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("before", array);
		
		Example3 e = new Example3(array);
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
			
