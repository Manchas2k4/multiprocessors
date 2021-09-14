public class Task2 extends Thread {
    private double result;
    private int n;

    public Task2(int n) {
        this.n = n;
        this.result = -1.0;
    }

    public double getResult() {
        return result;
    }

    public void run() {
        try {
            Thread.sleep(5000);
        } catch (InterruptedException ie) {
            ie.printStackTrace();
        }
        result = 1;
        for (int i = 1; i <= n; i++) {
            result *= i;
        }
    }
}
