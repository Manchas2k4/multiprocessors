import java.util.Arrays;


public class Example14 {
    private static final int SIZE = 100_000_000;

    public static double getAverage(int array[]) {
        double result = 0;
        for (int i = 0; i < array.length; i++) {
            result += array[i];
        }
        return result = result / array.length;
    }

    public static void main(String args[]) {
        int array[], temp[], mid = 0;
		long startTime, stopTime;
		double ms, avg = 0, median = 0;
        Mode thread1 = null;
        Variance thread2 = null;

		array = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("before", array);

        System.out.printf("Starting...\n");
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

            temp = Arrays.copyOf(array, array.length);
            Arrays.parallelSort(temp);
            avg = Example14.getAverage(temp);

            thread1 = new Mode(temp);
            thread2 = new Variance(temp, avg);

            thread1.start(); thread2.start();

            mid = array.length / 2;
            median = (array[mid] + array[mid + 1]) / 2.0;

			try {
				thread1.join(); thread2.join();
			} catch(InterruptedException e) {
				e.printStackTrace();
			}

			stopTime = System.currentTimeMillis();

			ms += (stopTime - startTime);
		}
        System.out.printf("avg = %.2f\n", avg);
        System.out.printf("median = %.2f\n", median);
        System.out.printf("mode = %d\n", thread1.getResult());
        System.out.printf("stddev = %.2f\n", thread2.getStdDev());
        System.out.printf("variance = %.2f\n", thread2.getVariance());
        System.out.printf("avg time = %.5f\n", (ms / Utils.N));
    }
}
