/* This code implements the known sort algorithm "Counting sort" */
import java.util.Arrays;

public class Example3 extends Thread {
	private int array[], temp[], start, end;;
	
	public Example3(int array[], int temp[], int start, int end) {
		this.array = array;
		this.temp = temp;
		this.start = start;
		this.end = end;
	}
	
	public void run() {
		int i, j, count;
		
		for (i = start; i < end; i++) {
			count = 0;
			for (j = 0; j < array.length; j++) {
				if (array[j] < array[i]) {
					count++;
				} else if (array[i] == array[j] && j < i) {
					count++;
				}
			}
			temp[count] = array[i];
		}
	}
}
			
