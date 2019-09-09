/* This code calculates de atan of x, x < 1 */

public class Example2 {
	private static final int LIMIT = 100_000_000;
	private double result, x;
	
	public Example2(double x) {
		this.x = x;
		this.result = 0;
	}
	
	public double getResult() {
		return result;
	}
	
	public void calculate() {
		double one, n;
		
		result = 0;
		for (int j = 0; j < LIMIT; j++) {
			one = (j % 2 == 0)? 1.0 : -1.0;
			n = (2 * j ) + 1;
			result += ( (one / n) * Math.pow(x, n) );
		}
	}
	
	public static void main(String args[]) {
		long startTime, stopTime;
		double acum = 0;
		
		Example2 e = new Example2(0.99);
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			e.calculate();
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		System.out.printf("arctan(0.99)->(0.78) = %f\n", e.getResult());
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
