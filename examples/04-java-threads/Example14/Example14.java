public class Example14 {
    public static void main(String args[]) {
        Task1 t1;
        Task2 t2;
        int array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        t1 = new Task1(array);
        t2 = new Task2(100);

        t1.start(); t2.start();

        try {
            t1.join(); t2.join();
        }

        System.out.println("t1 = " + t1.getResult());
        System.out.println("t2 = " + t2.getResult());
    }
}
