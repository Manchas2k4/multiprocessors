public class Intro04 {
    public static void main(String args[]) {
        Thread threads[];
        Counter counter;

        counter = new Counter();

        threads = new Thread[10];
        for (int i = 0; i < threads.length; i++) {
            if (i % 2 == 0) {
                threads[i] = new Increment(i, counter);
            } else {
                threads[i] = new Decrement(i, counter);
            }
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