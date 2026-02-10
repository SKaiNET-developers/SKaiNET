package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.TokenizerStrategy
import sk.ainet.apps.llm.TokenizerType

/**
 * Tokenizer strategy for SentencePiece models (LLaMA, Mistral, T5).
 * Uses ▁ (U+2581 LOWER ONE EIGHTH BLOCK) as the space marker.
 *
 * SentencePiece treats text as a sequence of Unicode characters and uses
 * the ▁ character to mark word boundaries (spaces become ▁, and text
 * starts with ▁ to mark the beginning).
 */
object SentencePieceStrategy : TokenizerStrategy {
    override val type: TokenizerType = TokenizerType.SENTENCEPIECE

    /** SentencePiece space marker: ▁ (U+2581) */
    override val spaceMarker: String = "\u2581"

    override fun preprocess(text: String): String {
        // SentencePiece adds space marker at start and replaces all spaces
        return spaceMarker + text.replace(" ", spaceMarker)
    }

    override fun postprocess(token: String): String {
        return when (token) {
            spaceMarker -> " "
            else -> token.replace(spaceMarker, " ")
        }
    }
}
