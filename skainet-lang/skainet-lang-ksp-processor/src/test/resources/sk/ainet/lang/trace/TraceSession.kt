package sk.ainet.lang.trace

/**
 * Mock TraceSession class for testing purposes.
 * This allows generated code to compile during tests.
 */
class TraceSession {
    private var nextId = 0
    
    fun refOf(tensor: Any): TensorRef {
        return TensorRef("t${nextId++}")
    }
    
    fun refsOf(tensors: List<Any>): List<TensorRef> {
        return tensors.map { refOf(it) }
    }
}