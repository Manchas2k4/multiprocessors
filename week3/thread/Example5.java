/* This code will generate a fractal image. */
import java.awt.image.BufferedImage;

public class Example5 extends Thread {
	private static final float SCALEX = 1.5f;
	private static final float SCALEY = 1.5f;
	private int array[], width, height, start, end;
	
	public Example5(int array[], int width, int height, int start, int end) {
		super();
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
	
	public int[] getArray() {
		return array;
	}
	
	public void run() {
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
}
