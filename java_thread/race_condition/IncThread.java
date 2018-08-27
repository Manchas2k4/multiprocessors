
public class IncThread extends Thread {
	private static final int MAXVAL = 50;
	private Counter myCounter;
	
	public IncThread(Counter counter) {
		myCounter = counter;
	}
	
	public void run() {
		for (int i = 0; i < MAXVAL; i++) {
			myCounter.increment();
			System.out.println("IncThread ID = " + this.getId() + 
					" value = " + myCounter.getValue());
		}
	}
}
