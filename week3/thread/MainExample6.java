/* This code will generate a fractal image. */
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class MainExample6 {
	private static final int MAXTHREADS = 4;
	
	public static void main(String args[]) throws Exception {
		Example6 threads[];
		int block;
		long startTime, stopTime;
		double acum = 0;
		
		if (args.length != 1) {
			System.out.println("usage: java Example6 image_file");
			System.exit(-1);
		}
		
		final String fileName = args[0];
		File srcFile = new File(fileName);
        final BufferedImage source = ImageIO.read(srcFile);
		
		int w = source.getWidth();
		int h = source.getHeight();
		int src[] = source.getRGB(0, 0, w, h, null, 0, w);
		int dest[] = new int[src.length];
		
		block = src.length / MAXTHREADS;
		threads = new Example6[MAXTHREADS];
		
		acum = 0;
		for (int j = 1; j <= Utils.N; j++) {
			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = new Example6(src, dest, w, h, (i * block), ((i + 1) * block));
				} else {
					threads[i] = new Example6(src, dest, w, h, (i * block), src.length);
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
		
		final BufferedImage destination = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
		destination.setRGB(0, 0, w, h, dest, 0, w);
		
		
		
		javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
               ImageFrame.showImage("Original - " + fileName, source);
            }
        });
		
		javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
               ImageFrame.showImage("Blur - " + fileName, destination);
            }
        });
	}
}