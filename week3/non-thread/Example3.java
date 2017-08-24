/* This code implements the known sort algorithm "Counting sort" */
import java.util.Arrays;

public class Example3 {
	private int array[];
	
	public Example3(int array[]) {
		this.array = array;
	}
	
	public int[] getArray() {
		return array;
	}
	
	public void doSort(boolean copy) {
		int temp[] = new int[array.length];
		int i, j, count;
		
		for (i = 0; i < array.length; i++) {
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
		if (copy) {
			array = Arrays.copyOf(temp, temp.length);
		}
	}
}
			
