public class Buffer {
	public static final int SIZE = 10;
	private char data[];
	private int index;
	
	public Buffer() {
		data = new char[SIZE];
		index = 0;
	}
	
	public synchronized void put(char c) {
		while (index == data.length) {
			try {
				wait();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		data[index++] = c;
		notify();
	}
	
	public synchronized char get() {
		char c;
		
		while (index == 0) {
			try {
				wait();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		c = data[--index];
		notify();
		return c;
	}
}
