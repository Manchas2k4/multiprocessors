public class Task1 extends Thread {
    private int array[];
    private int result;

    public Task1(int array[]) {
        this.array = array;
        this.result = -1;
    }

    public int getResult() {
        return result;
    }

    public void run() {
        try {
            Thread.sleep(5000);
        } catch (InterruptedException ie) {
            ie.printStackTrace();
        }
        result = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] % 5 == 0) {
                result++;
            }
        }
    }
}
