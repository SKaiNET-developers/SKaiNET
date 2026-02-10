package sk.ainet.apps.kllama

import sk.ainet.apps.llm.Tokenizer
import kotlinx.io.Source
import kotlinx.io.readByteArray
import kotlinx.io.readFloatLe
import kotlinx.io.readIntLe

class TokenizerUtils {
    companion object {
        fun buildTokenizer(
            source: Source,
            vocabSize: Int
        ): Tokenizer {
            val vocab = arrayOfNulls<String>(vocabSize)
            val vocabScores = FloatArray(vocabSize)
            val maxTokenLength = source.readIntLe()
            for (i in 0 until vocabSize) {
                vocabScores[i] = source.readFloatLe()
                val len = source.readIntLe()
                val bytes = source.readByteArray(len)
                vocab[i] = bytes.decodeToString()
            }
            return TokenizerImpl(vocab, vocabScores)
        }
    }
}