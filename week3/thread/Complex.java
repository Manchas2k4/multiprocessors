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