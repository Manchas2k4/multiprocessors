import java.util.Random;
import java.util.concurrent.BlockingQueue;

public class Producer extends Thread {
	public static Random r = new Random();
	private static int timeout = 100;
	private static int times = 10;
	private BlockingQueue<Character> buffer;
	
	public Producer(BlockingQueue<Character> buffer) {
		this.buffer = buffer;
	}
	
	public void run() {
		char c;
		
		System.out.println("Producer " + this.getId() + " started");
		for (int i = 0; i < times; i++) {
			try {
				buffer.put((char) (r.nextInt(26) + 'A'));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			try {
				Thread.sleep((r.nextInt(timeout) + 1) * 100);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			System.out.println("Producer " + this.getId() + " put a char");
		}
		System.out.println("Producer " + this.getId() + " has ended");
	}
}
