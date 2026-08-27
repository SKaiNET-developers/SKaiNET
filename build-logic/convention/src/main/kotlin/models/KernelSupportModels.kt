package models

import kotlinx.serialization.Serializable

/**
 * Machine-readable kernel × platform support, emitted by
 * `KernelSupportMatrixTest` (registry introspection) and rendered to an Antora
 * `.adoc` by `GenerateKernelMatrixTask` — the kernel-side analogue of
 * `operators.json` → `ops-status-matrix.adoc`.
 */
@Serializable
data class KernelSupportModule(
    val schema: String = "https://skainet.ai/schemas/kernel-support/v1",
    val version: String = "",
    val op: String = "matmul",
    val inputDtype: String = "Float32",
    val platforms: List<String> = emptyList(),
    val formats: List<KernelFormatSupport> = emptyList(),
    /**
     * Mapped serving (#1189): formats whose weight a kernel reads in canonical row-major GGUF
     * file order straight from off-heap (mmap'd/direct-buffer) bytes — no heap staging, no
     * prepack. Absent/empty in pre-#1189 JSON, so old files stay decodable.
     */
    val mapped: List<KernelFormatSupport> = emptyList(),
)

@Serializable
data class KernelFormatSupport(
    val name: String,
    /** platform name -> best provider serving `inputDtype × name` there (or absent = none). */
    val byPlatform: Map<String, String> = emptyMap(),
)
