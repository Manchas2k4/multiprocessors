// =================================================================
//
// File: Complex.java
// Author: Pedro Perez
// Description: This file contains the implementation of the 
//				Complex class used in Example 7.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Complex {
	private float real, img;
	
	public Complex(float r, float i) {
		real = r;
		img = i;
	}
	
	public float magnitude2() {
		return (real * real) + (img * img);
	}
	
	public Complex mult(Complex a) {
		return new Complex( ((real * a.real) - (img * a.img)),
		 				((img * a.real) + (real * a.img)) );
	}
	
	public Complex add(Complex a) {
		return new Complex( (real + a.real),
		 				(img + a.img) );
	}
}