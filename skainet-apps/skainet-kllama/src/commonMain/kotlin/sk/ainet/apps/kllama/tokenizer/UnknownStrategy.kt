package sk.ainet.apps.kllama.tokenizer

import sk.ainet.apps.kllama.TokenizerStrategy
import sk.ainet.apps.kllama.TokenizerType

/**
 * Fallback tokenizer strategy when the tokenizer type cannot be determined.
 * Defaults to SentencePiece-like behavior as it's most common in GGUF models.
 */
object UnknownStrategy : TokenizerStrategy {
    override val type: TokenizerType = TokenizerType.UNKNOWN

    /** Default to SentencePiece space marker: ▁ (U+2581) */
    override val spaceMarker: String = "\u2581"

    override fun preprocess(text: String): String {
        // Default to SentencePiece behavior
        return spaceMarker + text.replace(" ", spaceMarker)
    }

    override fun postprocess(token: String): String {
        return when (token) {
            spaceMarker -> " "
            else -> token.replace(spaceMarker, " ")
        }
    }
}
