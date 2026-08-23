package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.ExperimentalMemoryApi

/** No system properties here: the registry path is always on (override in tests via [DispatchMode.overrideEnabled]). */
@ExperimentalMemoryApi
internal actual fun platformUseRegistry(): Boolean = true
