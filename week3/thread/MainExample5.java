/* This code will generate a fractal image. */
import java.awt.image.BufferedImage;
import java.util.Arrays;

public class MainExample5 {
	private static final int WIDTH = 1024;
	private static final int HEIGHT = 768;
	private static final int MAXTHREADS = 4;
	
	public static void main(String args[]) {
		Example5 threads[];
		int block;
		long startTime, stopTime;
		double acum = 0;
		
		int array[] = new int[WIDTH * HEIGHT];
		
		block = array.length / MAXTHREADS;
		threads = new Example5[MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example5(array, WIDTH, HEIGHT, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example5(array, WIDTH, HEIGHT, (i * block), array.length);
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
