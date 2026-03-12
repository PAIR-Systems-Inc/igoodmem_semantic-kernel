package ai.goodmem.semantickernel;

/**
 * Thrown when the GoodMem REST API returns a non-2xx response or when
 * a client-side error occurs communicating with the server.
 */
public class GoodMemException extends RuntimeException {

    public GoodMemException(String message) {
        super(message);
    }

    public GoodMemException(String message, Throwable cause) {
        super(message, cause);
    }
}
