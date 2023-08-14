public class Variance extends Thread {
    private int array[];
    private double result, avg;

    public Variance(int array[], double avg) {
        this.array = array;
        this.avg = avg;
        this.result = 0.0;
    }

    public double getVariance() {
        return result;
    }

    public double getStdDev() {
        return Math.sqrt(result);
    }

    public void run() {
        result = 0;
        for (int i = 0; i < array.length; i++) {
            result += ((array[i] - avg)* (array[i] - avg));
        }
        result = result / array.length;
    }
}
