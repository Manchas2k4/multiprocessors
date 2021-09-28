// =================================================================
//
// File: Example9.java
// Author: Pedro Perez
// Description: This file implements the code  will generate a
//				fractal image using Java's Fork-Join.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.IOException;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example9 extends RecursiveAction {
	private static final int MIN = 10_000;
	private static final int WIDTH = 1920;
	private static final int HEIGHT = 1080;
	private static final float SCALEX = 0.5f;
	private static final float SCALEY = 0.5f;
	private static final int MAX_COLOR = 255;
	private static final float RED_PCT = 0.2f;
	private static final float GREEN_PCT = 0.4f;
	private static final float BLUE_PCT = 0.7f;
	private int array[], start, end;

	public Example9(int array[], int start, int end) {
		this.array = array;
		this.start = start;
		this.end = end;
	}

	private int juliaValue(int x, int y) {
		int k;
		float jx = SCALEX * (float) (WIDTH / 2 - x) / (WIDTH / 2);
		float jy = SCALEY * (float) (HEIGHT / 2 - y) / (HEIGHT / 2);
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
			ren = index / WIDTH;
			col = index % WIDTH;
			pixel = array[index];

			value = juliaValue(col, ren);

			r = (int) (MAX_COLOR * (RED_PCT * value));
			g = (int) (MAX_COLOR * (GREEN_PCT * value));
			b = (int) (MAX_COLOR * (BLUE_PCT * value));

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
			invokeAll(new Example9(array, start, mid),
					  new Example9(array, mid, end));
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		double ms;
		int array[];
		ForkJoinPool pool;

		array = new int[WIDTH * HEIGHT];

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			pool = new ForkJoinPool(Utils.MAXTHREADS);
			pool.invoke(new Example9(array, 0, array.length));

			stopTime = System.currentTimeMillis();
			ms += (stopTime - startTime);
		}

		System.out.printf("avg time = %.5f\n", (ms / Utils.N));

		final BufferedImage destination = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_ARGB);
		destination.setRGB(0, 0, WIDTH, HEIGHT, array, 0, WIDTH);

		/*
		javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
               ImageFrame.showImage("CPU Julia | c(-0.8, 0.156)", bi);
            }
        });
		*/

		try {
			ImageIO.write(destination, "png", new File("fractal.png"));
			System.out.println("Image was written succesfully.");
		} catch (IOException ioe) {
			System.out.println("Exception occured :" + ioe.getMessage());
		}
	}
}
