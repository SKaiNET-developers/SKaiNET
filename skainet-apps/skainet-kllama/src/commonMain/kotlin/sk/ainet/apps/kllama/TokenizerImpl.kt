package sk.ainet.apps.kllama

import sk.ainet.apps.llm.Tokenizer

class TokenizerImpl(
    private val vocab: Array<String?>,
    private val vocabScores: FloatArray,
) : Tokenizer {

    private val vocabSize = vocab.size
    // ----------------------------------------------------------------------------
    // byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt
    private fun strLookup(str: String, vocabSize: Int): Int {
        // find the first perfect match for str in vocab, return its index or -1 if not found
        for (i in 0 until vocabSize) {
            if (str == vocab[i]) {
                return i
            }
        }
        return -1
    }

    /**
     * Byte Pair Encoding
     * https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt
     */
    private fun bytePairEncodingEncode(text: String): IntArray {
        // first encode every individual byte in the input string
        val tokens = IntArray(text.length)
        var nTokens = 0 // the number of tokens
        for (element in text) {
            val singleChar = element.toString()
            val id = strLookup(singleChar, vocabSize)
            if (id == -1) {
                throw IllegalStateException("Failed to find $singleChar in vocab")
            }
            tokens[nTokens] = id
            nTokens++
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            var bestScore = -1e10f
            var bestId = -1
            var bestIdx = -1
            for (i in 0 until nTokens - 1) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                val strBuffer = vocab[tokens[i]] + vocab[tokens[i + 1]]
                val id = strLookup(strBuffer, vocabSize)
                if (id != -1 && vocabScores[id] > bestScore) {
                    // this merge pair exists in vocab! record its score and position
                    bestScore = vocabScores[id]
                    bestId = id
                    bestIdx = i
                }
            }
            if (bestIdx == -1) {
                break // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[bestIdx] = bestId
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (i in bestIdx + 1 until nTokens - 1) {
                tokens[i] = tokens[i + 1]
            }
            nTokens-- // token length decreased
        }
        return tokens.copyOfRange(0, nTokens)
    }

    override fun encode(text: String): IntArray {
        return bytePairEncodingEncode(text)
    }

    override fun decode(token: Int): String {
        // following BOS token (1), sentence piece decoder strips any leading whitespace (see PR#89)
        try {
            if (vocab[token] == null) {
                return ""
            }
            val tokenStr = if (token == 1 && vocab[token]!![0] == ' ') {

                vocab[token]!!.substring(1)
            } else {
                vocab[token]!!
            }
            return tokenStr
        } catch (e: Exception) {
            throw IllegalStateException("Failed to find token $token in vocab")
        }
    }

    override fun decode(tokens: IntArray): String {
        return tokens.joinToString(separator = "") { decode(it) }
    }

}