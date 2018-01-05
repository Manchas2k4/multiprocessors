/* This code adds all the values of an array */
import java.util.concurrent.RecursiveAction;

public class Example2 extends RecursiveAction {
	private static final long MIN = 100_000;
	private int a[], b[], c[], start, end;
	
	public Example2(int c[], int a[], int b[], int start, int end) {
		this.a = a;
		this.b = b;
		this.c = c;
		this.start = start;
		this.end = end;
	}
	
	protected void computeDirectly() {
		for (int i = this.start; i < this.end; i++) {
			c[i] = a[i] + b[i];
		}
	}
	
	@Override
	protected void compute() {
		if ( (this.end - this.start) <= Example2.MIN ) {
			computeDirectly();
			
		} else {
			int middle = (end + start) / 2;
			
			invokeAll(new Example2(c, b, a, start, middle), 
					  new Example2(c, b, a, middle, end));
		}
		
	}
}
			
