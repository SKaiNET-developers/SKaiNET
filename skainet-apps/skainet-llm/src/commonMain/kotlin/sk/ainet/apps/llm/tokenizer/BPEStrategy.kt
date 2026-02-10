package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.TokenizerStrategy
import sk.ainet.apps.llm.TokenizerType

/**
 * Tokenizer strategy for GPT-2/GPT-3 style BPE models.
 * Uses Ġ (U+0120 LATIN CAPITAL LETTER G WITH DOT ABOVE) as the space marker.
 *
 * GPT-2 BPE tokenizers encode spaces as part of the following token,
 * using Ġ to represent a space before the token.
 */
object BPEStrategy : TokenizerStrategy {
    override val type: TokenizerType = TokenizerType.BPE

    /** GPT-2 BPE space marker: Ġ (U+0120) */
    override val spaceMarker: String = "\u0120"

    override fun preprocess(text: String): String {
        // GPT-2 style: space becomes Ġ prefix on following token
        // First token doesn't get a space prefix unless the text starts with space
        return text.replace(" ", spaceMarker)
    }

    override fun postprocess(token: String): String {
        return when (token) {
            spaceMarker -> " "
            else -> token.replace(spaceMarker, " ")
        }
    }
}
