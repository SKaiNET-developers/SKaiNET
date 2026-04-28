package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.KernelProvider

/**
 * `ServiceLoader`-friendly wrappers around the singleton kernel
 * providers shipped in this module. `ServiceLoader` requires a public
 * no-arg constructor, which Kotlin `object` declarations don't expose.
 * Each wrapper is a regular class that delegates to the singleton via
 * `KernelProvider by <singleton>`, so all calls — `name`, `priority`,
 * `isAvailable()`, `matmulFp32()` — route directly back to the object.
 *
 * The wrappers themselves carry no state and aren't meant to be used
 * directly by application code; depend on
 * [ScalarKernelProvider] / [PanamaVectorKernelProvider] instead. The
 * only consumer is the JVM `ServiceLoader` machinery driven by
 * [sk.ainet.backend.api.kernel.KernelServiceLoader].
 */
public class ScalarKernelProviderFactory : KernelProvider by ScalarKernelProvider

public class PanamaVectorKernelProviderFactory : KernelProvider by PanamaVectorKernelProvider
