public class Intro03 extends Thread {
    private int id, start, end;

    public Intro03(int id, int start, int end) {
        this.id = id;
        this.start = start;
        this.end = end;
    }

    public void run() {
        for (int i = start; i < end; i++) {
            System.out.println("id = " + id + " i = " + i);
        }
    }

    public static void main(String arr[]) {
        Intro03 threads[];
        int max = 20;

        threads = new Intro03[5];

        for (int i = 0; i < threads.length; i++) {
            threads[i] = new Intro03(i, (i * max), ((i + 1) * max));
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