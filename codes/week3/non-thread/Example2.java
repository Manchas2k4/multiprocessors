/* This code adds all the values of an array */

public class Example2 {
	private int a[], b[], c[];
	
	public Example2(int c[], int a[], int b[]) {
		this.a = a;
		this.b = b;
		this.c = c;
	}
	
	public void calculate() {
		for (int i = 0; i < c.length; i++) {
			c[i] = a[i] + b[i];
		}
	}
}
			
