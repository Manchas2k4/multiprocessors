public class Intro02 extends Thread {
    private int id, limit;

    public Intro02(int id, int limit) {
        this.id = id;
        this.limit = limit;
    }

    public void run() {
        for (int i = 0; i < limit; i++) {
            System.out.println("id = " + id + " i = " + i);
        }
    }

    public static void main(String arr[]) {
        Intro02 threads[];

        threads = new Intro02[10];

        for (int i = 0; i < threads.length; i++) {
            threads[i] = new Intro02(i, 20);
            threads[i].start();
        }

        for (int i = 0; i < threads.length; i++) {
            try {
                threads[i].join();
            } catch (InterruptedException ie) {
                ie.printStackTrace();
            }
        }
    }
}