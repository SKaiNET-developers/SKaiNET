package sk.ainet.compile.minerva

/**
 * Minerva graph export support for secure MCU inference.
 *
 * The phase-one implementation is JVM-first and targets static sequential MLP
 * graphs with Q8 libminerva compilation. Host verification is intentionally a
 * separate stage so compiler packaging can be tested without MCU hardware.
 */
public object MinervaExportBackend {
    public const val backendName: String = "minerva"
    public const val phaseOneScope: String = "jvm-sequential-mlp-q8"
}
