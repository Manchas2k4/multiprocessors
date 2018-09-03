import java.util.Random;

public class Producer extends Thread {
	private static Random r = new Random();
	private int factor, timeout, loop;
	
	public Producer(int f, int t, int l) {
		factor = f;
		timeout = t;
		loop = l;
	}
	
	public void espera() {
		try {
			Thread.sleep((r.nextInt(factor) + 1) * timeout);
		} catch (InterruptedException ie) {
			ie.printStackTrace();
		}
	}
	
	public void run() {
		char c;
		
		System.out.println("Producer " + getId() + " started");
		for (int i = 0; i < loop; i++) {
			c = (char) (r.nextInt(26) + 'A');
			System.out.println("Producer " + getId() + " put " + c);
			Buffer.getInstance().put(c);
			espera();
		}
		System.out.println("Producer " + getId() + " finished");
	}
}
