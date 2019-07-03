public class Parallel extends Thread {
	private int array[], low, high;
	private long result;
	
	public Parallel(int array[], int low, int high) {
		this.array = array;
		this.low = low;
		this.high = high;
		this.result = 0;
	}
	
	public long getResult() {
		return result;
	}
	
	private int maximumSum(int array[], int low, int mid, int high) {
		int i, left, acum, right;
		
		left = acum = 0;
		for (i = mid; i >= low; i--) {
			acum += array[i];
			left = Math.max(left, acum);
		}
		right = acum = 0;
		for (i = mid + 1; i < high; i++) {
			acum += array[i];
			right = Math.max(right, acum);
		}
		return (left + right);
	}
	
	public void run() {
		if ( (high - low + 1) == 1 ) {
			result = array[low];
		} else {
			int mid = low + ( (high - low) / 2 );
			Parallel left = new Parallel(array, low, mid);
			Parallel right = new Parallel(array, mid + 1, high);
			
			left.start(); right.start();
			int center = maximumSum(array, low, mid, high);
			
			try {
				left.join(); right.join();
			} catch (InterruptedException ie) {
				ie.printStackTrace();
			}
			result = Math.max(center, Math.max(left.getResult(), right.getResult()));	
		}
	}
}
