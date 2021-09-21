import java.util.Hashtable;
import java.util.Iterator;

public class Mode extends Thread {
    private int array[];
    private int result;
    private Hashtable<Integer, Integer> counters;

    public Mode(int array[]) {
        this.array = array;
        this.result = 0;
        this.counters = new Hashtable<Integer, Integer>();
    }

    public int getResult() {
        return result;
    }

    public void run() {
        for (int i = 0; i < array.length; i++) {
            if (!counters.containsKey(array[i])) {
                counters.put(array[i], 1);
            } else {
                counters.put(array[i], counters.get(array[i]) + 1);
            }
        }

        for (Iterator<Integer> itr = counters.keySet().iterator(); itr.hasNext(); ) {
            result = Math.max(counters.get(itr.next()), result);
        }
    }
}
