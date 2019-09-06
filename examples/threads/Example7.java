/* This code will generate a fractal image. */
import java.awt.image.BufferedImage;

public class Example7 extends Thread {
	private static final float SCALEX = 1.5f;
	private static final float SCALEY = 1.5f;
	private int array[], width, height, start, end;
	
	public Example7(int array[], int width, int height, int start, int end) {
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
	
	public static void main(String args[]) {
		final int WIDTH = 1024;
		final int HEIGHT = 768;
		Example7 threads[];
		int block;
		long startTime, stopTime;
		double acum = 0;
		
		int array[] = new int[WIDTH * HEIGHT];
		
		block = array.length / Utils.MAXTHREADS;
		threads = new Example7[Utils.MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example7(array, WIDTH, HEIGHT, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example7(array, WIDTH, HEIGHT, (i * block), array.length);
				}
			}
			
			startTime = System.currentTimeMillis();
			for (int i = 0; i < threads.length; i++) {
				threads[i].start();
			}
			for (int i = 0; i < threads.length; i++) {
				try {
					threads[i].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			stopTime = System.currentTimeMillis();
			acum +=  (stopTime - startTime);
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
