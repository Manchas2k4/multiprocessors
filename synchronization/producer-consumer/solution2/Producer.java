import java.util.Random;
import java.util.concurrent.BlockingQueue;

public class Producer extends Thread {
	private static Random r = new Random();
	private int factor, timeout, loop;
	private BlockingQueue<Character> myBuffer;
	
	public Producer(int f, int t, int l, BlockingQueue<Character> buffer) {
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
		
		System.out.println("Producer " + getId() + " started");
		for (int i = 0; i < loop; i++) {
			c = (char) (r.nextInt(26) + 'A');
			System.out.println("Producer " + getId() + " put " + c);
			try {
				myBuffer.put(c);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			espera();
		}
		System.out.println("Producer " + getId() + " finished");
	}
}
