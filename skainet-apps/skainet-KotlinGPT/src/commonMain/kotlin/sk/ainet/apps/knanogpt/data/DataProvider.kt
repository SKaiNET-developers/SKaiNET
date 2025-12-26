package sk.ainet.apps.knanogpt.data

interface DataProvider<T> {
    fun load(): T
}