package sk.ainet.lang.memory

/** No property store here: debug mode is opt-in through [MemoryDebug.overrideEnabled]. */
@ExperimentalMemoryApi
internal actual fun platformMemoryDebugEnabled(): Boolean = false

/** No cheap stack walk on this target. */
@ExperimentalMemoryApi
internal actual fun platformCallSite(): String? = null
