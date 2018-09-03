import java.util.Random;

public class Consumer extends Thread {
	private static Random r = new Random();
	private int factor, timeout, loop;
	
	public Consumer(int f, int t, int l) {
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
		
		System.out.println("\t\tConsumer " + getId() + " started");
		for (int i = 0; i < loop; i++) {
			c = Buffer.getInstance().get();
			System.out.println("\t\tConsumer " + getId() + " get " + c);
			espera();
		}
		System.out.println("\t\tConsumer " + getId() + " finished");
	}
}
