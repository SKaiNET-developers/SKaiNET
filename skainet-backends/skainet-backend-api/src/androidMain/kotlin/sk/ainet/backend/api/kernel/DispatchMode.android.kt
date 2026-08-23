package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.ExperimentalMemoryApi

@ExperimentalMemoryApi
internal actual fun platformUseRegistry(): Boolean = System.getProperty(DispatchMode.PROPERTY) != "false"
