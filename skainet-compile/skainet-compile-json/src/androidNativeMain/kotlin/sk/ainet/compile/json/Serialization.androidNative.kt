package sk.ainet.compile.json

import sk.ainet.compile.json.model.SkJsonExport

/**
 * One file for both `androidNativeArm32` and `androidNativeArm64` — the default hierarchy
 * template groups them under `androidNativeMain`, so this does not need the per-target
 * duplication the other native actuals here carry.
 */
public actual suspend fun writeExportToFile(export: SkJsonExport, path: String, pretty: Boolean) {
    throw NotImplementedError("writeExportToFile is not implemented for Android native in this module. Use toJsonString() and handle file I/O in the host app.")
}
