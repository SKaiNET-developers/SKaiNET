package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.TokenizerStrategy
import sk.ainet.apps.llm.TokenizerType

/**
 * Tokenizer strategy for WordPiece models (BERT).
 * Uses ## as the continuation marker for subword tokens.
 *
 * WordPiece first splits on whitespace, then breaks words into subwords.
 * The first subword of each word has no prefix, subsequent subwords
 * are prefixed with ##.
 *
 * Example: "unbelievable" might become ["un", "##believ", "##able"]
 */
object WordPieceStrategy : TokenizerStrategy {
    override val type: TokenizerType = TokenizerType.WORDPIECE

    /** WordPiece continuation marker: ## */
    override val spaceMarker: String = "##"

    override fun preprocess(text: String): String {
        // WordPiece doesn't transform text the same way - it splits on whitespace
        // and uses ## for continuations. The BPE loop handles this differently.
        // For now, we just return the text as-is since word splitting
        // happens at a different stage.
        return text
    }

    override fun postprocess(token: String): String {
        // Remove ## prefix from continuation tokens
        return if (token.startsWith("##")) {
            token.substring(2)
        } else {
            token
        }
    }
}
