/* This code will generate a fractal image. */
import java.awt.image.BufferedImage;

public class Example7 {
	private static final int WIDTH = 1024;
	private static final int HEIGHT = 768;
	private static final float SCALEX = 1.5f;
	private static final float SCALEY = 1.5f;
	private int array[], width, height;
	
	public Example7(int array[], int width, int height) {
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
	
	void doMagic() {
		int index, ren, col, value, pixel, r, g, b;
		
		for (index = 0; index < array.length; index++) {
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
		long startTime, stopTime;
		double acum = 0;
		
		int array[] = new int[WIDTH * HEIGHT];
		Example7 e = new Example7(array, WIDTH, HEIGHT);
		acum = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();
			e.doMagic();
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
