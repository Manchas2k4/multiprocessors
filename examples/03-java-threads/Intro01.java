public class Intro01 extends Thread {
    private int limit;

    public Intro01(int limit) {
        this.limit = limit;
    }

    public void run() {
        for (int i = 0; i < limit; i++) {
            System.out.println("i = " + i);
        }
    }

    public static void main(String arr[]) {
        Intro01 thread;

        thread = new Intro01(10000);

        thread.start();

        try {
            thread.join();
        } catch (InterruptedException ie) {
            ie.printStackTrace();
        }
    }
}