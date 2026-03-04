@file:JvmName("LlamaIngestionBlocking")

package sk.ainet.apps.kllama.java

import kotlinx.coroutines.runBlocking
import kotlinx.io.Source
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.io.RandomAccessSource
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.types.DType

/**
 * Blocking (non-suspend) wrappers for LlamaIngestion loading methods.
 *
 * These are JVM-only extension functions that wrap the suspend functions
 * in runBlocking, making them callable from plain Java code.
 */

/**
 * Blocking version of [LlamaIngestion.load].
 * Loads LLaMA weights from a GGUF source synchronously.
 */
public fun <T : DType> LlamaIngestion<T>.loadBlocking(sourceProvider: () -> Source): LlamaRuntimeWeights<T> =
    runBlocking { load(sourceProvider) }

/**
 * Blocking version of [LlamaIngestion.loadStreaming].
 * Loads LLaMA weights using streaming API synchronously.
 */
public fun <T : DType> LlamaIngestion<T>.loadStreamingBlocking(
    randomAccessProvider: () -> RandomAccessSource
): LlamaRuntimeWeights<T> = runBlocking { loadStreaming(randomAccessProvider) }
