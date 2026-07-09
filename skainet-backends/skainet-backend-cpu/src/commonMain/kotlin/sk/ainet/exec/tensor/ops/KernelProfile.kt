package sk.ainet.exec.tensor.ops

import kotlin.time.TimeSource

/**
 * Lightweight, always-on accumulating profiler for the matmul dispatch paths.
 * Diagnostic only — used to localize where native decode time goes (quant-NEON
 * vs FP32-scalar vs generic) before investing in a kernel rewrite. The clock
 * read per call is negligible next to a matmul. Read [report] after a run and
 * [reset] between phases (e.g. to separate prefill from decode).
 */
public object KernelProfile {
    private val clock = TimeSource.Monotonic

    public var quantNanos: Long = 0; private set
    public var quantCalls: Long = 0; private set
    public var fp32Nanos: Long = 0; private set
    public var fp32Calls: Long = 0; private set
    public var genericNanos: Long = 0; private set
    public var genericCalls: Long = 0; private set

    public fun <R> timeQuant(body: () -> R): R {
        val mark = clock.markNow(); val r = body()
        quantNanos += mark.elapsedNow().inWholeNanoseconds; quantCalls++; return r
    }

    public fun <R> timeFp32(body: () -> R): R {
        val mark = clock.markNow(); val r = body()
        fp32Nanos += mark.elapsedNow().inWholeNanoseconds; fp32Calls++; return r
    }

    public fun <R> timeGeneric(body: () -> R): R {
        val mark = clock.markNow(); val r = body()
        genericNanos += mark.elapsedNow().inWholeNanoseconds; genericCalls++; return r
    }

    public fun reset() {
        quantNanos = 0; quantCalls = 0
        fp32Nanos = 0; fp32Calls = 0
        genericNanos = 0; genericCalls = 0
    }

    public fun report(): String {
        fun ms(ns: Long) = ns / 1_000_000.0
        val total = quantNanos + fp32Nanos + genericNanos
        fun pct(ns: Long) = if (total > 0) 100.0 * ns / total else 0.0
        return buildString {
            appendLine("[KernelProfile] matmul time breakdown:")
            appendLine("  quant-NEON   : ${ms(quantNanos)} ms over $quantCalls calls (${pct(quantNanos)}%)")
            appendLine("  fp32-scalar  : ${ms(fp32Nanos)} ms over $fp32Calls calls (${pct(fp32Nanos)}%)")
            appendLine("  generic      : ${ms(genericNanos)} ms over $genericCalls calls (${pct(genericNanos)}%)")
            append("  matmul total : ${ms(total)} ms")
        }
    }
}
