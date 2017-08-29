/* This code adds all the values of an array */

public class Example1 extends Thread {
	private int array[], start, end;
	private long result;
	
	public Example1(int array[], int start, int end) {
		super();
		this.array = array;
		this.start = start;
		this.end = end;
		this.result = 0;
	}
	
	public long getResult() {
		return result;
	}
	
	public void run() {
		result = 0;
		for (int i = start; i < end; i++) {
			result += array[i];
		}
	}
}
			
