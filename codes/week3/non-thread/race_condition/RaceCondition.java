
public class RaceCondition {
	public static void main(String args[]) {
		Counter c;
		IncThread inc1, inc2;
		DecThread dec1, dec2;
		
		c = new Counter();
		inc1 = new IncThread(c);
		inc2 = new IncThread(c);
		dec1 = new DecThread(c);
		dec2 = new DecThread(c);
		
		inc1.start();
		inc2.start();
		dec1.start();
		dec2.start();
	}
}
