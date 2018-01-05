
public class DecThread extends Thread {
	private static final int MAXVAL = 10;
	private Counter myCounter;
	
	public DecThread(Counter counter) {
		myCounter = counter;
	}
	
	public void run() {
		for (int i = 0; i < MAXVAL; i++) {
			myCounter.decrement();
			System.out.println("DecThread ID = " + this.getId() + 
					" value = " + myCounter.getValue());
		}
	}
}
