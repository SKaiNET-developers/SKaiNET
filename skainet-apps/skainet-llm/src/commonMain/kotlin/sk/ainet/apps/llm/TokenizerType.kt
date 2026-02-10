package sk.ainet.apps.llm

/**
 * Supported tokenizer types based on their preprocessing strategies.
 */
enum class TokenizerType {
    /** LLaMA, Mistral, T5 - uses ▁ (U+2581) as space marker */
    SENTENCEPIECE,

    /** GPT-2, GPT-3 - uses Ġ (U+0120) as space marker */
    BPE,

    /** BERT - uses ## for continuation tokens */
    WORDPIECE,

    /** Fallback when type cannot be determined */
    UNKNOWN
}

/**
 * Strategy interface for tokenizer preprocessing and postprocessing.
 * Different tokenizer types handle whitespace and word boundaries differently.
 */
interface TokenizerStrategy {
    /**
     * The type of tokenizer this strategy implements.
     */
    val type: TokenizerType

    /**
     * The character/string used to represent spaces in the vocabulary.
     */
    val spaceMarker: String

    /**
     * Preprocess input text before BPE encoding.
     * This typically involves handling whitespace and word boundaries.
     */
    fun preprocess(text: String): String

    /**
     * Postprocess a decoded token to restore original text.
     * This typically involves converting space markers back to spaces.
     */
    fun postprocess(token: String): String
}
