/* This code adds all the values of an array */

public class Example4 {
	private int a[], b[], c[];
	
	public Example4(int c[], int a[], int b[]) {
		this.a = a;
		this.b = b;
		this.c = c;
	}
	
	public void calculate() {
		for (int i = 0; i < c.length; i++) {
			c[i] = a[i] + b[i];
		}
	}
	
	public static void main(String args[]) {
		final int SIZE = 100_000_000;
		long startTime, stopTime;
		double acum = 0;
		
		int a[] = new int[SIZE];
		Utils.fillArray(a);
		Utils.displayArray("a", a);
		
		int b[] = new int[SIZE];
		Utils.fillArray(b);
		Utils.displayArray("b", b);
		
		int c[] = new int[SIZE];
		
		Example4 e = new Example4(c, a, b);
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			e.calculate();
			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		Utils.displayArray("c", c);
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
	}
}
			
