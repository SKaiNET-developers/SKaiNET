package sk.ainet.apps.knanogpt.transformer

data class TransformerConfig(
    val head_size: Int,
    val n_embd: Int,
    val num_heads: Int,
    val dropout: Double,
    val vocab_size: Int,
    val block_size: Int,
    val n_layer: Int)
