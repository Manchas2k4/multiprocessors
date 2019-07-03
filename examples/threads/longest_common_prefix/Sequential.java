public class Sequential {
	private String findPrefix(char a[], char b[]) {
		int i, j;
		String result;
		
		result = "";
		i = j = 0;
		while (i < a.length && j < b.length) {
			if (a[i] != b[i]) {
				break;
			}
			result = result + a[i];
			i++; j++;
		}
		return result;
	}
	
	private int recursiveTask(String array[], int low, int high) {
		if ( (high - low + 1) == 1 ) {
			return array[low];
		}
		
		int mid = low + ( (high - low) / 2 );
		String left = recursiveTask(array, low, mid);
		String right = recursiveTask(array, mid + 1, high);
		return findPrefix(left, right);
	}
	
	public int doTask(int array[]) {
		return recursiveTask(array, 0, array.length - 1);
	}
}
