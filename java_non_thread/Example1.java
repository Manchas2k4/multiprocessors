/* This code adds all the values of an array */

public class Example1 {
	private int array[];
	private long result;
	
	public Example1(int array[]) {
		this.array = array;
		this.result = 0;
	}
	
	public long getResult() {
		return result;
	}
	
	public void calculate() {
		result = 0;
		for (int i = 0; i < array.length; i++) {
			result += array[i];
		}
	}
}
			
