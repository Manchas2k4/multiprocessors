/* This code will generate a fractal image. */
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;

public class MainExample5 {
	private static final int WIDTH = 1024;
	private static final int HEIGHT = 768;
	private static final int MAXTHREADS = Runtime.getRuntime().availableProcessors();
	
	public static void main(String args[]) {
		ForkJoinPool pool;
		long startTime, stopTime;
		double acum = 0;
		
		int array[] = new int[WIDTH * HEIGHT];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			startTime = System.currentTimeMillis();
			
			pool = new ForkJoinPool(MAXTHREADS);
			pool.invoke(new Example5(array, WIDTH, HEIGHT, 0, array.length));
			
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
