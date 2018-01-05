import java.util.Random;
import java.util.concurrent.BlockingQueue;

public class Consumer extends Thread {
	public static Random r = new Random();
	private static int timeout = 100;
	private static int times = 10;
	private BlockingQueue<Character> buffer;
	
	public Consumer(BlockingQueue<Character> buffer) {
		this.buffer = buffer;
	}
	
	public void run() {
		char c;
		
		System.out.println("\t\tConsumer " + this.getId() + " started");
		for (int i = 0; i < times; i++) {
			c = ' ';
			try {
				c = buffer.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			try {
				Thread.sleep((r.nextInt(timeout) + 1) * 100);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			System.out.println("\t\tConsumer " + this.getId() + " got " + c);
		}
		System.out.println("\t\tConsumer " + this.getId() + " has ended");
	}
}
