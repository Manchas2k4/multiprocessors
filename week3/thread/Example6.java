/* This code will generate a fractal image. */
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class Example6 extends Thread{
	private static final int BLUR_WINDOW = 15;
	private int src[], dest[], width, height, start, end;
	
	public Example6(int src[], int dest[], int width, int height, int start, int end) {
		super();
		this.src = src;
		this.dest = dest;
		this.width = width;
		this.height = height;
		this.start = start;
		this.end = end;
	}
	
	private void blur_pixel(int ren, int col) {
		int side_pixels, i, j, cells;
		int tmp_ren, tmp_col, pixel, dpixel;
		float r, g, b;
	
		side_pixels = (BLUR_WINDOW - 1) / 2;
		cells = (BLUR_WINDOW * BLUR_WINDOW);
		r = 0; g = 0; b = 0;
		for (i = -side_pixels; i <= side_pixels; i++) {
			for (j = -side_pixels; j <= side_pixels; j++) {
				tmp_ren = Math.min( Math.max(ren + i, 0), height - 1 );
				tmp_col = Math.min( Math.max(col + j, 0), width - 1);
				pixel = src[(tmp_ren * width) + tmp_col];
			
				r += (float) ((pixel & 0x00ff0000) >> 16);
				g += (float) ((pixel & 0x0000ff00) >> 8);
				b += (float) ((pixel & 0x000000ff) >> 0);
			}
		}
	
		dpixel = (0xff000000)
				| (((int) (r / cells)) << 16)
				| (((int) (g / cells)) << 8)
				| (((int) (b / cells)) << 0);
		dest[(ren * width) + col] = dpixel;
	}
	
	public void run() {
		int index, size;
		int ren, col;
		
		size = width * height;
		for (index = start; index < end; index++) {
			ren = index / width;
			col = index % width;
			blur_pixel(ren, col);
		}
	}
}