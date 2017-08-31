public class Main {
	public static void main(String args[]) {
		Thread threads[] = new Thread[6];
		Buffer buffer = new Buffer();
		
		for (int i = 0; i < 3; i++) {
			threads[i] = new Consumer(buffer);
			threads[i + 3] = new Producer(buffer);
		}
		
		int i;
		for (i = 0; i < 3; i++) {
			threads[i].start();
		}
		
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		for (; i < 6; i++) {
			threads[i].start();
		}
	}
}
