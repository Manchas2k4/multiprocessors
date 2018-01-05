import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

/**
 * Starts the web server and listens to client connections. Individual 
 * connections are processed by Worker instances.
 */
public class WebServer {

    public static final int PORT_NUMBER = 12345;

    private void listen() {        
        System.out.printf("Minuscule web server ready. Listening at port %d...%n", 
                PORT_NUMBER);
        
        try (
                ServerSocket servSock = new ServerSocket(PORT_NUMBER)
            ) {
            
            while (true) {
                Socket sock = servSock.accept();
                Worker w = new Worker(sock);
                (new MyThread(w)).start();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        new WebServer().listen();
    }
}
