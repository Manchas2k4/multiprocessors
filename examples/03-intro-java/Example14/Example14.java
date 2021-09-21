import java.util.Arrays;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Map.Entry;

public class Example14 {
    private static final int SIZE = 100_000_000;
    private int array[];
    private double avg, variance, median;
    private int mode;

    public Example14(int array[]) {
        this.array = array;
        this.avg = 0;
        this.variance = 0;
        this.median = 0;
        this.mode = 0;
    }

    public void calculateAverage() {
        avg = 0;
        for (int i = 0; i < array.length; i++) {
            avg += array[i];
        }
        avg = avg / array.length;
    }

    public void calculateMode() {
        Hashtable<Integer, Integer> counters;

        counters = new Hashtable<Integer, Integer>();
        for (int i = 0; i < array.length; i++) {
            if (!counters.containsKey(array[i])) {
                counters.put(array[i], 1);
            } else {
                counters.put(array[i], counters.get(array[i]) + 1);
            }
        }

        Entry<Integer, Integer> max = null;
        for (Iterator<Entry<Integer, Integer>> itr = counters.entrySet().iterator(); itr.hasNext(); ) {
            Entry<Integer, Integer> current = itr.next();
            if ( (max == null) || (current.getValue() > max.getValue()) ) {
                max = current;
            }
        }
        mode = max.getKey();
    }

    public void calculateVariance() {
        variance = 0;
        for (int i = 0; i < array.length; i++) {
            variance += ((array[i] - avg)* (array[i] - avg));
        }
        variance = variance / array.length;
    }

    public double getAvg() {
        return avg;
    }

    public double getMedian() {
        return median;
    }

    public int getMode() {
        return mode;
    }

    public double getStdDev() {
        return Math.sqrt(variance);
    }

    public double getVariance() {
        return variance;
    }

    public void doTask() {
        int mid = array.length / 2;

        Arrays.sort(array);
        calculateAverage();
        median = (array[mid] + array[mid + 1]) / 2.0;
        calculateMode();
        calculateVariance();
    }

    public static void main(String args[]) {
        int array[];
		long startTime, stopTime;
		double ms;
        Example14 obj = null;

		array = new int[SIZE];
		Utils.randomArray(array);
		Utils.displayArray("before", array);

        Arrays.parallelSort(array);

        System.out.printf("Starting...\n");
		ms = 0;
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

            obj = new Example14(Arrays.copyOf(array, array.length));
            obj.doTask();

			stopTime = System.currentTimeMillis();

			ms += (stopTime - startTime);
		}
        System.out.printf("avg = %.2f\n", obj.getAvg());
        System.out.printf("median = %.2f\n", obj.getMedian());
        System.out.printf("mode = %d\n", obj.getMode());
        System.out.printf("stddev = %.2f\n", obj.getStdDev());
        System.out.printf("variance = %.2f\n", obj.getVariance());
        System.out.printf("avg time = %.5f\n", (ms / Utils.N));
    }
}
