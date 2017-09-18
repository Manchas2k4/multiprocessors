import java.util.LinkedList;
import java.net.Socket;

public class PoolThreads {
	private static final int MAX = 2;
	private final LinkedList<MyThread> threads;
	
	public PoolThreads() {
		threads = new LinkedList<MyThread>();
		
		for (int i = 0; i < MAX; i++) {
			MyThread aux = new MyThread(this);
			aux.start();
			threads.add(aux);
		}
	}
	
	public synchronized void assignJob(Socket socket) {
		if (threads.isEmpty()) {
			BusyWorker b = new BusyWorker(socket);
			b.doWork();
		} else {
			MyThread t = threads.removeFirst();
			t.add(new Worker(socket));
		}
	}
	
	public synchronized void addThread(MyThread t) {
		threads.add(t);
	}
}
