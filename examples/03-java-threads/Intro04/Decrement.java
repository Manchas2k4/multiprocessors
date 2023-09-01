public class Decrement extends Thread {
    private Counter c;
    private int id;

    public Decrement(int id, Counter c) {
        this.id = id;
        this.c = c;
    }

    public void run() {
        for (int i = 0; i < 10; i++) {
            c.count--;
            System.out.println("Decrement id = " + id + " count = " + c.count);
        }
    }
}