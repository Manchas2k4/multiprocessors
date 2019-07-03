import java.util.Random;

public class Utils {
	public static final int N = 10;
	private static final int DISPLAY = 100;
	private static final int MAX_VALUE = 10_000;	
	private static final Random random = new Random();
	
	public static int[] getRandomArray(int size) throws IllegalArgumentException {
		if (size < 1) {
			throw new IllegalArgumentException();
		}
		
		int array[] = new int[size];
		for (int i = 0; i < size; i++) {
			array[i] = random.nextInt() % MAX_VALUE;
		}
		
		return array;
	}
	
	public static int[] getRandomPositiveArray(int size) throws IllegalArgumentException {
		if (size < 1) {
			throw new IllegalArgumentException();
		}
		
		int array[] = new int[size];
		for (int i = 0; i < size; i++) {
			array[i] = random.nextInt(MAX_VALUE) + 1;
		}
		
		return array;
	}
	
	public static int[] getRandomIncrementalArray(int size) throws IllegalArgumentException {
		if (size < 1) {
			throw new IllegalArgumentException();
		}
		
		int array[] = new int[size];
		for (int i = 0; i < size; i++) {
			array[i] = (i % MAX_VALUE) + 1;
		}
		
		return array;
	}
	
	public static void displayArray(String text, int array[]) {
		System.out.printf("%s = [%4d", text, array[0]);
		for (int i = 1; i < DISPLAY; i++) {
			System.out.printf(",%4d", array[i]);
		}
		System.out.printf(", ..., ]\n");
	}
}
