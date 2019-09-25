/* This code will generate a fractal image. */
import java.awt.image.BufferedImage;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example7 extends RecursiveAction {
	private static final int WIDTH = 1024; // 4096;
	private static final int HEIGHT = 768; //3112;
	private static final float SCALEX = 1.5f;
	private static final float SCALEY = 1.5f;
	private static final int MIN = 10_000;
	private int array[], width, height, start, end;
	
	public Example7(int start, int end, int array[], int width, int height) {
		this.start = start;
		this.end = end;
		this.array = array;
		this.width = width;
		this.height = height;
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

	protected void computeDirectly() {
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
		if ( (end - start) <= MIN ) {
			computeDirectly();
		} else {
			int mid = start + ((end - start) / 2);
			invokeAll(new Example7(start, mid, array, width, height),
					  new Example7(mid, end, array, width, height));
		}
	}
	
	public static void main(String args[]) {
		long startTime, stopTime;
		double acum = 0;
		ForkJoinPool pool;
		
		int array[] = new int[WIDTH * HEIGHT];
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			pool.invoke(new Example7(0, WIDTH * HEIGHT, array, WIDTH, HEIGHT));

			stopTime = System.currentTimeMillis();
			acum += (stopTime - startTime);
		}
		System.out.printf("avg time = %.5f\n", (acum / Utils.N));
		
		final BufferedImage bi = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_ARGB);
		bi.setRGB(0, 0, WIDTH, HEIGHT, array, 0, WIDTH);
		javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
               ImageFrame.showImage("CPU Julia | c(-0.8, 0.156)", bi);
            }
        });
	}
}
