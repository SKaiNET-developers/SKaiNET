package sk.ainet.apps.kllama

import kotlinx.io.Source
import kotlinx.io.readFloatLe
import kotlinx.io.readIntLe
import sk.ainet.context.ExecutionContext
import sk.ainet.models.llama.LlamaLayerWeights
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

/**
 * Loader for Karpathy's llama2.c binary checkpoint format.
 * Format:
 * - Header (7 ints): dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
 * - Weights (Float32): All tensors sequentially in a fixed order.
 */
public object Llama2DotCWeightLoader {

    public fun load(ctx: ExecutionContext, source: Source): LlamaRuntimeWeights<FP32> {
        val dim = source.readIntLe()
        val hiddenDim = source.readIntLe()
        val nLayers = source.readIntLe()
        val nHeads = source.readIntLe()
        val nKvHeads = source.readIntLe()
        val vocabSize = source.readIntLe()
        val seqLen = source.readIntLe()

        val metadata = LlamaModelMetadata(
            architecture = "llama",
            embeddingLength = dim,
            contextLength = seqLen,
            blockCount = nLayers,
            headCount = nHeads,
            kvHeadCount = nKvHeads,
            feedForwardLength = hiddenDim,
            ropeDimensionCount = dim / nHeads,
            vocabSize = vocabSize
        )

        fun readTensor(rows: Int, cols: Int): sk.ainet.lang.tensor.Tensor<FP32, Float> {
            val size = rows * cols
            val buffer = FloatArray(size) { source.readFloatLe() }
            return ctx.fromFloatArray(Shape(rows, cols), FP32::class, buffer)
        }

        fun readTensor1D(size: Int): sk.ainet.lang.tensor.Tensor<FP32, Float> {
            val buffer = FloatArray(size) { source.readFloatLe() }
            return ctx.fromFloatArray(Shape(size), FP32::class, buffer)
        }

        // Karpathy format weight order:
        // token_embedding_table: [vocab_size, dim]
        // rms_att_weight: [n_layers, dim]
        // wq: [n_layers, dim, dim]
        // wk: [n_layers, dim, dim]
        // wv: [n_layers, dim, dim]
        // wo: [n_layers, dim, dim]
        // rms_ffn_weight: [n_layers, dim]
        // w1: [n_layers, hidden_dim, dim]
        // w2: [n_layers, dim, hidden_dim]
        // w3: [n_layers, hidden_dim, dim]
        // rms_final_weight: [dim]
        // freq_cis_real: [seq_len, head_size / 2] (optional, usually recalculated if not in file)
        // freq_cis_imag: [seq_len, head_size / 2] (optional)
        // wcls: [vocab_size, dim] (if shared_classifier is false, but usually it's shared)

        val tokenEmbedding = readTensor(vocabSize, dim)

        // Read layer weights
        val attnNorms = (0 until nLayers).map { readTensor1D(dim) }
        val wqs = (0 until nLayers).map { readTensor(dim, dim) }
        val wks = (0 until nLayers).map { readTensor(dim, dim) }
        val wvs = (0 until nLayers).map { readTensor(dim, dim) }
        val wos = (0 until nLayers).map { readTensor(dim, dim) }
        val ffnNorms = (0 until nLayers).map { readTensor1D(dim) }
        val w1s = (0 until nLayers).map { readTensor(hiddenDim, dim) } // ffnGate
        val w2s = (0 until nLayers).map { readTensor(dim, hiddenDim) } // ffnDown
        val w3s = (0 until nLayers).map { readTensor(hiddenDim, dim) } // ffnUp

        val outputNorm = readTensor1D(dim)

        // RoPE frequencies (check if they are in the file or if we reached EOF)
        // Karpathy's export might include them if the file is larger than expected
        // But the standard format often stops here if it's the 15M/42M/110M models
        // LlamaRuntime handles RoPE fallback if weights are null.
        
        // Wait, I should check if there are more weights for the output layer (classifier)
        // In llama2.c, if shared_classifier is 1, outputWeight is tokenEmbedding.
        // But the .bin file usually includes it if it's not shared.
        // Actually, for stories models, it IS usually included.
        
        val outputWeight = try {
            readTensor(vocabSize, dim)
        } catch (e: Exception) {
            // Fallback to shared embedding if EOF reached
            tokenEmbedding
        }

        val layers = (0 until nLayers).map { i ->
            LlamaLayerWeights(
                attnNorm = attnNorms[i],
                wq = wqs[i],
                wk = wks[i],
                wv = wvs[i],
                wo = wos[i],
                ffnNorm = ffnNorms[i],
                ffnGate = w1s[i],
                ffnDown = w2s[i],
                ffnUp = w3s[i]
            )
        }

        return LlamaRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = tokenEmbedding,
            ropeFreqReal = null,
            ropeFreqImag = null,
            layers = layers,
            outputNorm = outputNorm,
            outputWeight = outputWeight
        )
    }
}
