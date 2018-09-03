public class Solution1Main {
	public static final int PRODUCER_FACTOR = 20;
	public static final int PRODUCER_TIMEOUT = 250;
	public static final int PRODUCER_LOOP = 20;
	
	public static final int CONSUMER_FACTOR = 20;
	public static final int CONSUMER_TIMEOUT = 250;
	public static final int CONSUMER_LOOP = 20;
	
	public static void main(String args[]) {
		for (int i = 0; i < 20; i++) {
			(new Producer(PRODUCER_FACTOR, PRODUCER_TIMEOUT, PRODUCER_LOOP)).start();
		}
		
		try {
			Thread.sleep(5000);
		} catch (InterruptedException ie) {
			ie.printStackTrace();
		}
		
		for (int i = 0; i < 20; i++) {
			(new Consumer(CONSUMER_FACTOR, CONSUMER_TIMEOUT, CONSUMER_LOOP)).start();
		}
	}
}

