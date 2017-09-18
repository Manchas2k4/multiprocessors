public class MyThread extends Thread {
	private final PoolThreads myPool;
	private volatile Worker myWorker;
	
	public MyThread(PoolThreads pool) {
		myPool = pool;
		myWorker = null;
	}
	
	public synchronized void add(Worker w) {
		myWorker = w;
		notify();
	}
	
	public synchronized void run() {
		while (true) {
			while (myWorker == null) {
				try {
					wait();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			myWorker.doWork();
			myPool.addThread(this);
			myWorker = null;
		}
	}
}
