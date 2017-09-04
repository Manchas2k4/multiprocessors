public class MyThread extends Thread {
	private Worker worker;
	
	public MyThread(Worker worker) {
		this.worker = worker;
	}
	
	public void run() {
		worker.doWork();
	}
}
