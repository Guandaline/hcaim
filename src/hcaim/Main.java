package hcaim;

import java.io.IOException;

/**
 *
 * @author guandaline
 */
public class Main {

    /**
     * @param args the command line arguments -i--<in-train-file> -o--<out-train-file> -r--<in-test-file> -s--<out-test-file>
     */
    public static void main(String[] args) throws IOException {
       Discretizer discretizer = new Discretizer(args);
       discretizer.run();
    }

}
