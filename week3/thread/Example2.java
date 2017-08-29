/* This code adds all the values of an array */

public class Example2 extends Thread {
	private int a[], b[], c[], start, end;
	
	public Example2(int c[], int a[], int b[], int start, int end) {
		this.a = a;
		this.b = b;
		this.c = c;
		this.start = start;
		this.end = end;
	}
	
	public void run() {
		for (int i = start; i < end; i++) {
			c[i] = a[i] + b[i];
		}
	}
}
			
