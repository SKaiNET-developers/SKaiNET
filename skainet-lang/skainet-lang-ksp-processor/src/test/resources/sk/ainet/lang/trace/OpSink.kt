package sk.ainet.lang.trace

/**
 * Mock OpSink interface for testing purposes.
 * This allows generated code to compile during tests.
 */
interface OpSink {
    fun onOpExecuted(trace: OpTrace)
}