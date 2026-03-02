package sk.ainet.apps.kllama.java

/**
 * Configuration for text generation, used by KLlamaJava facades.
 *
 * Example usage from Java:
 * ```java
 * GenerationConfig config = GenerationConfig.builder()
 *     .maxTokens(256)
 *     .temperature(0.7f)
 *     .build();
 * ```
 */
public class GenerationConfig private constructor(
    public val maxTokens: Int,
    public val temperature: Float
) {
    public companion object {
        @JvmStatic
        public fun builder(): Builder = Builder()

        /** Default configuration: 256 max tokens, temperature 0.8. */
        @JvmStatic
        public fun defaults(): GenerationConfig = Builder().build()
    }

    public class Builder {
        private var maxTokens: Int = 256
        private var temperature: Float = 0.8f

        public fun maxTokens(maxTokens: Int): Builder {
            this.maxTokens = maxTokens
            return this
        }

        public fun temperature(temperature: Float): Builder {
            this.temperature = temperature
            return this
        }

        public fun build(): GenerationConfig = GenerationConfig(maxTokens, temperature)
    }
}
