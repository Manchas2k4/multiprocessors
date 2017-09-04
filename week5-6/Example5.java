/* This code will generate a fractal image. */
import java.awt.image.BufferedImage;
import java.util.concurrent.RecursiveAction;

public class Example5 extends RecursiveAction {
	private static final float SCALEX = 1.5f;
	private static final float SCALEY = 1.5f;
	private static final long MIN = 10_000;
	private int array[], width, height, start, end;
	
	public Example5(int array[], int width, int height, int start, int end) {
		this.array = array;
		this.width = width;
		this.height = height;
		this.start = start;
		this.end = end;
	}
	
	private int juliaValue(int x, int y) {
		int k;
		float jx = SCALEX * (float) (width / 2 - x) / (width / 2);
		float jy = SCALEY * (float) (height / 2 - y) / (height / 2);
		Complex c = new Complex(-0.8f, 0.156f);
		Complex a = new Complex(jx, jy);
	 
		for (k = 0; k < 200; k++) {
		    a = (a.mult(a)).add(c);
		    if (a.magnitude2() > 1000) {
		        return 0;
		    }
		}
		return 1;
	}
	
	public void computeDirectly() {
		int index, ren, col, value, pixel, r, g, b;
		
		for (index = start; index < end; index++) {
			ren = index / width;
			col = index % width;
			pixel = array[index];
			
			value = juliaValue(col, ren);
			
			r = (int) (255 * (0.4 * value));
			g = (int) (255 * (0.5 * value));
			b = (int) (255 * (0.7 * value));
			
			array[index] =  (0xff000000)
							| (((int) r) << 16)
							| (((int) g) << 8)
							| (((int) b) << 0);
		}
	}
	
	@Override
	protected void compute() {
		if ( (this.end - this.start) <= Example5.MIN ) {
			computeDirectly();
			
		} else {
			int middle = (end + start) / 2;
			
			invokeAll(new Example5(array, width, height, start, middle), 
					  new Example5(array, width, height, middle, end));
		}
	}
}
