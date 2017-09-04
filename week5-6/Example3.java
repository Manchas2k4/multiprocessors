/* This code implements the known sort algorithm "Counting sort" */
import java.util.Arrays;
import java.util.concurrent.RecursiveAction;

public class Example3 extends RecursiveAction {
	private static final long MIN = 1_000;
	private int array[], temp[], start, end;
	
	public Example3(int array[], int temp[], int start, int end) {
		this.array = array;
		this.temp = temp;
		this.start = start;
		this.end = end;
	}
	
	protected void computeDirectly() {
		int count;
		
		for (int i = start; i < end; i++) {
			count = 0;
			for (int j = 0; j < array.length; j++) {
				if (array[j] < array[i]) {
					count++;
				} else if (array[i] == array[j] && j < i) {
					count++;
				}
			}
			temp[count] = array[i];
		}
	}
	
	@Override
	protected void compute() {
		if ( (this.end - this.start) <= Example3.MIN ) {
			computeDirectly();
			
		} else {
			int middle = (end + start) / 2;
			
			invokeAll(new Example3(array, temp, start, middle), 
					  new Example3(array, temp, middle, end));
		}
		
	}
}
			
