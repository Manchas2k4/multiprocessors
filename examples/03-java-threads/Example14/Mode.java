import java.util.Hashtable;
import java.util.Iterator;
import java.util.Map.Entry;

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

        Entry<Integer, Integer> max = null;
        for (Iterator<Entry<Integer, Integer>> itr = counters.entrySet().iterator(); itr.hasNext(); ) {
            Entry<Integer, Integer> current = itr.next();
            if ( (max == null) || (current.getValue() > max.getValue()) ) {
                max = current;
            }
        }
        result = max.getKey();
    }
}
