import java.util.Random;

public class Producer extends Thread {
	private static Random r = new Random();
	private static int timeout = 100;
	private static int times = 10;
	private Buffer buffer;
	
	public Producer(Buffer buffer) {
		this.buffer = buffer;
	}
	
	public void run() {
		char c;
		
		System.out.println("Producer " + this.getId() + " started");
		for (int i = 0; i < times; i++) {
			buffer.put((char) (r.nextInt(26) + 'A'));
			System.out.println("Producer " + this.getId() + " put a char");
			try {
				Thread.sleep((r.nextInt(timeout) + 1) * 100);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("Producer " + this.getId() + " has ended");
	}
}
