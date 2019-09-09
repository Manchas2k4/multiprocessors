/* This code adds all the values of an array */

public class Example1 {
	private static final long NUM_RECTS = 1_000_000_000;
	private double result;
	
	public Example1() {
		this.result = 0;
	}
	
	public double getResult() {
		return result;
	}
	
	public void calculate() {
		double sum, width, mid, height;
		
		sum = 0;
		width = 1.0 / (double) NUM_RECTS;
		for (int j = 0; j < NUM_RECTS; j++) {
			mid = (j + 0.5) * width;
			height = 4.0 / (1.0 + (mid * mid));
			sum += height;
		}
		result = width * sum;
	}
	
	public static void main(String args[]) {
		long startTime, stopTime;
		double acum = 0;
		
		Example1 e = new Example1();
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			e.calculate();
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		System.out.printf("PI = %d\n", e.getResult());
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
