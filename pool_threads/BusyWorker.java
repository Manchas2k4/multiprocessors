import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.net.Socket;

/**
 * Processes a client connection. Reads the request information from a socket 
 * and produces a simple response.
 */
public class BusyWorker {

    public static final String EOL = "\r\n";
    private Socket sock;

    public BusyWorker(Socket sock) {
        this.sock = sock;
    }
    
    public void doWork() {

        String content = "The server is currently unavailable";

        try (
                InputStream input = sock.getInputStream();
                PrintStream output = new PrintStream(sock.getOutputStream())
            ) {

            System.out.println("-------------------------------------");            
            System.out.println(readInputBytes(input));
            
            output.print("HTTP/1.0 503 Service Unavailable" + EOL);
            output.print("Content-type: text/plain" + EOL);
            output.print("Content-length: " + content.length() + EOL);
            output.print(EOL);
            output.print(content);

        } catch (IOException e) {
            e.printStackTrace();

        } finally {
            try {
                sock.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    
    // A utility function.
    private static String readInputBytes(InputStream input) {

        try {
            while (true) {
                int n = input.available();
                if (n == 0) {
                    Thread.sleep(50);
                    continue;
                }
                byte[] bytes = new byte[n];
                input.read(bytes);
                return new String(bytes);
            }

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return null;
        }
    }
}
