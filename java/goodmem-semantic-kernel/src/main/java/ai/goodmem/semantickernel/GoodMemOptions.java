package ai.goodmem.semantickernel;

import java.util.Objects;

/**
 * Configuration for the GoodMem Semantic Kernel connector.
 * All settings have defaults sourced from {@code GOODMEM_*} environment variables.
 *
 * <p>Use the builder to override specific values:
 * <pre>{@code
 * GoodMemOptions options = GoodMemOptions.builder()
 *     .baseUrl("http://localhost:8080")
 *     .apiKey("my-key")
 *     .build();
 * }</pre>
 */
public final class GoodMemOptions {

    private final String baseUrl;
    private final String apiKey;
    private final String embedderId;
    private final boolean verifySsl;

    private GoodMemOptions(Builder b) {
        this.baseUrl    = b.baseUrl;
        this.apiKey     = b.apiKey;
        this.embedderId = b.embedderId;
        this.verifySsl  = b.verifySsl;
    }

    /** Base URL of the GoodMem server (no trailing slash). */
    public String getBaseUrl()    { return baseUrl; }

    /** API key sent in the {@code x-api-key} header. */
    public String getApiKey()     { return apiKey; }

    /**
     * Embedder UUID to use when creating spaces.
     * When {@code null} the first available embedder is used automatically.
     */
    public String getEmbedderId() { return embedderId; }

    /**
     * Whether to validate TLS certificates.
     * Set to {@code false} for local servers with self-signed certificates.
     */
    public boolean isVerifySsl()  { return verifySsl; }

    /** Returns a new builder initialised from environment variables. */
    public static Builder builder() { return new Builder(); }

    public static final class Builder {

        private String baseUrl    = env("GOODMEM_BASE_URL",    "http://localhost:8080");
        private String apiKey     = env("GOODMEM_API_KEY",     "");
        private String embedderId = System.getenv("GOODMEM_EMBEDDER_ID"); // nullable
        private boolean verifySsl = !"false".equalsIgnoreCase(System.getenv("GOODMEM_VERIFY_SSL"));

        private Builder() {}

        public Builder baseUrl(String baseUrl) {
            this.baseUrl = Objects.requireNonNull(baseUrl, "baseUrl");
            return this;
        }

        public Builder apiKey(String apiKey) {
            this.apiKey = Objects.requireNonNull(apiKey, "apiKey");
            return this;
        }

        /** Nullable — pass {@code null} to auto-select the first available embedder. */
        public Builder embedderId(String embedderId) {
            this.embedderId = embedderId;
            return this;
        }

        public Builder verifySsl(boolean verifySsl) {
            this.verifySsl = verifySsl;
            return this;
        }

        public GoodMemOptions build() {
            return new GoodMemOptions(this);
        }

        private static String env(String key, String defaultValue) {
            String val = System.getenv(key);
            return (val != null && !val.isEmpty()) ? val : defaultValue;
        }
    }
}
