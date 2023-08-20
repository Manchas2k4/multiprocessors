// =================================================================
//
// File: Example07.java
// Author: Pedro Perez
// Description: This file implements the code that blurs a given
//				image using Java's Threads.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.IOException;

public class Example07 extends Thread {
	private static final int BLUR_WINDOW = 15;
	private int src[], dest[], width, height, start, end;

	public Example07(int start, int end, int src[], int dest[], int width, int height) {
		this.start = start;
		this.end = end;
		this.src = src;
		this.dest = dest;
		this.width = width;
		this.height = height;
	}

	private void blurPixel(int ren, int col) {
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
			blurPixel(ren, col);
		}
	}

	public static void main(String args[]) throws Exception {
		long startTime, stopTime;
		double elapsedTime;
		int blockSize;
		Example07 threads[];

		if (args.length != 1) {
			System.out.println("usage: java Example10 image_file");
			System.exit(-1);
		}

		final String fileName = args[0];
		File srcFile = new File(fileName);
        final BufferedImage source = ImageIO.read(srcFile);

		int w = source.getWidth();
		int h = source.getHeight();
		int src[] = source.getRGB(0, 0, w, h, null, 0, w);
		int dest[] = new int[src.length];

		int size = source.getWidth() * source.getHeight();
		blockSize = size / Utils.MAXTHREADS;
		threads = new Example07[Utils.MAXTHREADS];

		System.out.printf("Starting...\n");
		elapsedTime = 0;
		for (int j = 0; j < Utils.N; j++) {
			startTime = System.currentTimeMillis();

			for (int i = 0; i < threads.length; i++) {
				if (i != threads.length - 1) {
					threads[i] = 
					new Example07((i * blockSize), ((i + 1) * blockSize), src, dest, w, h);
				} else {
					threads[i] = 
					new Example07((i * blockSize), size, src, dest, w, h);
				}
			}

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

			elapsedTime += (stopTime - startTime);
		}
		System.out.printf("avg time = %.5f\n", (elapsedTime / Utils.N));
		final BufferedImage destination = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
		destination.setRGB(0, 0, w, h, dest, 0, w);

		/*
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
		*/

		try {
			ImageIO.write(destination, "png", new File("blur.png"));
			System.out.println("Image was written succesfully.");
		} catch (IOException ioe) {
			System.out.println("Exception occured :" + ioe.getMessage());
		}
	}
}
