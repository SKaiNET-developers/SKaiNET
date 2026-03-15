package sk.ainet.compile.json

import sk.ainet.compile.json.model.SkJsonExport

public actual suspend fun writeExportToFile(export: SkJsonExport, path: String, pretty: Boolean) {
    throw NotImplementedError("writeExportToFile is not supported on Wasm WASI. Use toJsonString() and transfer via WASI I/O.")
}
