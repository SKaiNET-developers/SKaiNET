package sk.ainet.compile.minerva

/**
 * Minerva graph export support for secure MCU inference.
 *
 * The first implementation slice is intentionally JVM-first and API-only: it
 * defines the SKaiNET-facing export surface and result model before the
 * validator, lowering, compiler adapter, packager, and host verifier are added.
 */
public object MinervaExportBackend {
    public const val backendName: String = "minerva"
    public const val phaseOneScope: String = "jvm-sequential-mlp-q8"
}
