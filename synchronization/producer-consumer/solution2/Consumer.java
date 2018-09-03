import java.util.Random;
import java.util.concurrent.BlockingQueue;

public class Consumer extends Thread {
	private static Random r = new Random();
	private int factor, timeout, loop;
	private BlockingQueue<Character> myBuffer;
	
	public Consumer(int f, int t, int l, BlockingQueue<Character> buffer) {
		factor = f;
		timeout = t;
		loop = l;
		myBuffer = buffer;
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
			try {
				c = myBuffer.take();
			} catch (InterruptedException e) {
				c = ' ';
				e.printStackTrace();
			}
			System.out.println("\t\tConsumer " + getId() + " get " + c);
			espera();
		}
		System.out.println("\t\tConsumer " + getId() + " finished");
	}
}
