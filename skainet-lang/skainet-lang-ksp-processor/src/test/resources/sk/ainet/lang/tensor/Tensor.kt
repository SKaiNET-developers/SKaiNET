package sk.ainet.lang.tensor

/**
 * Mock Tensor interface for testing purposes.
 * This allows generated code to compile during tests.
 */
interface Tensor<T, V> {
    val shape: List<Int>
    val data: V
}