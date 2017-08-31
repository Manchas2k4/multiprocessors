import java.util.Random;

public class Consumer extends Thread {
	private static Random r = new Random();
	private static int timeout = 100;
	private static int times = 10;
	private Buffer buffer;
	
	public Consumer(Buffer buffer) {
		this.buffer = buffer;
	}
	
	public void run() {
		char c;
		
		System.out.println("\t\tConsumer " + this.getId() + " started");
		for (int i = 0; i < times; i++) {
			c = buffer.get();
			System.out.println("\t\t Consumer " + this.getId() + " got " + c);
			try {
				Thread.sleep((r.nextInt(timeout) + 1) * 100);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("\t\tConsumer " + this.getId() + " has ended");
	}
}
