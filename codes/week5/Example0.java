/* This code adds all the values of an array */
import java.util.concurrent.RecursiveTask;

public class Example0 extends RecursiveTask<Double> {
	private static final long MIN = 100_000;
	private int start, end;
	private double width;
	
	public Example0(int start, int end, double width ) {
		this.start = start;
		this.end = end;
		this.width = width;
	}
	
	protected Double computeDirectly() {
		double mid, height;
		double sum = 0;
		
		for (int i = start; i < end; i++) {
			mid = (i + 0.5) * width;
			height = 4.0 / (1.0 + (mid * mid));
			sum += height;
		}
		return (width * sum);
	}
	
	@Override
	protected Double compute() {
		if ( (this.end - this.start) <= Example0.MIN ) {
			return this.computeDirectly();
			
		} else {
			int mid = (end + start) / 2;
			Example0 lowerMid = new Example0(start, mid, width);
			lowerMid.fork();
			Example0 upperMid = new Example0(mid, end, width);
			return (upperMid.compute() + lowerMid.join());
		}
	}
}
