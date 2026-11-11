package sk.ainet.io.safetensors

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.openRandomAccessSource
import sk.ainet.io.model.LoadingProgress
import sk.ainet.io.model.LoadingStage
import sk.ainet.io.model.ProgressReportingLoader
import sk.ainet.lang.tensor.storage.TensorStorage

/**
 * Streaming reader for sharded/multi-file SafeTensors models.
 *
 * Reads metadata from all shards without loading tensor data into memory.
 * Emits progress events via [progress] flow during loading.
 *
 * Sharded SafeTensors models consist of:
 * - An index file: `model.safetensors.index.json`
 * - Multiple shard files: `model-00001-of-00003.safetensors`, etc.
 *
 * Usage:
 * ```kotlin
 * // Collect progress in a coroutine
 * val progressJob = scope.launch {
 *     StreamingShardedSafeTensorsReader.progress.collect { progress ->
 *         updateUI(progress)
 *     }
 * }
 *
 * // Open from index file
 * val reader = StreamingShardedSafeTensorsReader.openFromIndex("/path/to/model.safetensors.index.json")
 *
 * // Use reader
 * println("Total tensors: ${reader.tensors.size}")
 * println("Shards loaded: ${reader.loadedShards.size}/${reader.index.shardCount}")
 *
 * progressJob.cancel()
 * ```
 *
 * Where there is no file system — a browser, a Wasm host — the shards come from wherever the caller
 * keeps them, through [openFromIndexContent]: the index as text and one [RandomAccessSource] per
 * shard name (a `JsBlobRandomAccessSource` over a `File` input, a `ByteArrayRandomAccessSource` over
 * fetched bytes). Both entry points open every shard exactly once and keep it open until [close].
 */
public class StreamingShardedSafeTensorsReader private constructor(
    /** The parsed index file */
    public val index: SafeTensorsIndex,
    /** Opens the shard with this file name, or returns `null` when it is missing. */
    private val openShard: suspend (shardFilename: String) -> RandomAccessSource?,
    /** The file path a shard name resolves to, when the shards are files; `null` otherwise. */
    private val pathOf: (shardFilename: String) -> String?,
) : ProgressReportingLoader, AutoCloseable {

    private val _progress = MutableSharedFlow<LoadingProgress>(replay = 0, extraBufferCapacity = 64)
    override val progress: Flow<LoadingProgress> = _progress

    /** All tensor metadata aggregated from all shards */
    private val _tensors: MutableList<ShardedTensorInfo> = mutableListOf()
    public val tensors: List<ShardedTensorInfo> get() = _tensors

    /** Metadata aggregated from all shards */
    private val _metadata: MutableMap<String, String> = mutableMapOf()
    public val metadata: Map<String, String> get() = _metadata

    /** List of successfully loaded shards */
    private val _loadedShards: MutableList<String> = mutableListOf()
    public val loadedShards: List<String> get() = _loadedShards

    /** List of missing/failed shards */
    private val _missingShards: MutableList<String> = mutableListOf()
    public val missingShards: List<String> get() = _missingShards

    /** Map of shard filename to its reader (for tensor data access) */
    private val shardReaders: MutableMap<String, StreamingSafeTensorsReader> = mutableMapOf()

    /** The file behind each loaded shard, when it has one ([pathOf], else the source's own `filePath`). */
    private val shardPaths: MutableMap<String, String?> = mutableMapOf()

    /** Whether all shards were loaded successfully */
    public val isComplete: Boolean get() = _missingShards.isEmpty()

    /** Total size from index metadata */
    public val totalSize: Long? get() = index.metadata.totalSize

    /**
     * Load tensor data by name.
     *
     * @param name The tensor name
     * @return Raw bytes for the tensor
     * @throws IllegalArgumentException if tensor not found
     * @throws IllegalStateException if the containing shard was not loaded
     */
    public fun loadTensorData(name: String): ByteArray {
        val tensor = _tensors.firstOrNull { it.name == name }
            ?: throw IllegalArgumentException("Tensor not found: $name")
        return loadTensorData(tensor)
    }

    /**
     * Load tensor data for a specific tensor.
     *
     * @param tensor The tensor info from [tensors] list
     * @return Raw bytes for the tensor
     * @throws IllegalStateException if the containing shard was not loaded
     */
    public fun loadTensorData(tensor: ShardedTensorInfo): ByteArray {
        val reader = shardReaders[tensor.shardFilename]
            ?: throw IllegalStateException("Shard not loaded: ${tensor.shardFilename}")
        return reader.loadTensorData(tensor.name)
    }

    /**
     * Same shape as [loadTensorData] but returns a file-backed
     * [TensorStorage] instead of a heap [ByteArray]. Lets callers
     * memory-map the tensor's byte range straight from the shard file
     * without going through a 2 GB-capped `ByteArray` round-trip.
     *
     * The returned [TensorStorage] holds a
     * [sk.ainet.lang.tensor.storage.BufferHandle.FileBacked] that
     * references the shard file by absolute path; callers (or the
     * runtime that consumes the storage) own the mmap lifecycle.
     *
     * Sharded analog of
     * [StreamingSafeTensorsReader.loadTensorStorageMapped]. The
     * shard's file path is resolved internally from the index — the
     * caller doesn't need to know which physical file contains the
     * tensor.
     *
     * @param tensor The tensor info from [tensors].
     * @return [TensorStorage] descriptor with a file-backed buffer
     *   handle pointing at the shard file's tensor byte range.
     * @throws IllegalStateException if the containing shard was not
     *   loaded, or if the per-shard reader does not surface the
     *   tensor (consistency check).
     * @throws UnsupportedOperationException if the shard did not come
     *   from a file (a Blob or in-memory source has nothing to map).
     */
    public fun loadTensorStorageMapped(tensor: ShardedTensorInfo): TensorStorage {
        val reader = shardReaders[tensor.shardFilename]
            ?: throw IllegalStateException("Shard not loaded: ${tensor.shardFilename}")
        val streamingTensor = reader.tensors.firstOrNull { it.name == tensor.name }
            ?: throw IllegalStateException(
                "Tensor '${tensor.name}' not found in shard '${tensor.shardFilename}'",
            )
        val path = shardPaths[tensor.shardFilename]
            ?: throw UnsupportedOperationException(
                "Shard '${tensor.shardFilename}' is not a file (its source has no filePath): mapped loading needs file-backed shards",
            )
        return reader.loadTensorStorageMapped(streamingTensor, path)
    }

    /**
     * Convenience overload for [loadTensorStorageMapped] that looks up
     * the tensor by name. Mirrors the [loadTensorData] name-based
     * overload.
     *
     * @throws IllegalArgumentException if no tensor matches [name].
     */
    public fun loadTensorStorageMapped(name: String): TensorStorage {
        val tensor = _tensors.firstOrNull { it.name == name }
            ?: throw IllegalArgumentException("Tensor not found: $name")
        return loadTensorStorageMapped(tensor)
    }

    override fun close() {
        shardReaders.values.forEach { it.close() }
        shardReaders.clear()
    }

    // ========== Internal Loading ==========

    private suspend fun loadShards(allowPartial: Boolean) {
        val shardFiles = index.shardFiles
        val totalStages = 2 + shardFiles.size + 1 // parse_index + discover + N shards + aggregate

        val startTime = currentTimeMillis()
        _progress.emit(LoadingProgress.Started(totalStages, "Loading sharded SafeTensors model"))

        // Stage: discover_shards
        val discoverStage = LoadingStage("discover_shards", "Discovering shards", 1, totalStages)
        _progress.emit(LoadingProgress.StageStarted(discoverStage, "Checking shard files"))
        val discoverStart = currentTimeMillis()

        // Each shard is opened once, here, and the source is handed to its reader below; a shard
        // whose opener answers `null` (or throws, under allowPartial) is missing.
        val openedShards = LinkedHashMap<String, RandomAccessSource>()
        val missingShardFiles = mutableListOf<String>()

        for (shardFilename in shardFiles) {
            val source = try {
                openShard(shardFilename)
            } catch (e: Exception) {
                if (!allowPartial) {
                    openedShards.values.forEach { it.close() }
                    _progress.emit(LoadingProgress.Failed(discoverStage, e, "Failed to open shard: $shardFilename"))
                    throw e
                }
                null
            }
            if (source != null) {
                openedShards[shardFilename] = source
            } else {
                missingShardFiles.add(shardFilename)
            }
        }

        _missingShards.addAll(missingShardFiles)

        if (missingShardFiles.isNotEmpty() && !allowPartial) {
            openedShards.values.forEach { it.close() }
            val incomplete = SafeTensorsShardException.IncompleteShard(
                openedShards.size,
                shardFiles.size,
                missingShardFiles
            )
            _progress.emit(LoadingProgress.Failed(discoverStage, incomplete, "Missing shards: ${missingShardFiles.joinToString()}"))
            throw incomplete
        }

        _progress.emit(LoadingProgress.StageCompleted(discoverStage, currentTimeMillis() - discoverStart))

        // Load each shard
        val availableShards = openedShards.keys.toList()
        for ((shardIndex, shardFilename) in availableShards.withIndex()) {
            val stageId = "load_shard_${shardIndex + 1}"
            val stageName = "Loading shard ${shardIndex + 1}/${availableShards.size}"
            val stage = LoadingStage(stageId, stageName, 2 + shardIndex, totalStages)

            _progress.emit(LoadingProgress.StageStarted(stage, shardFilename))
            val shardStart = currentTimeMillis()

            val source = openedShards.getValue(shardFilename)
            try {
                val reader = StreamingSafeTensorsReader.open(source)
                shardReaders[shardFilename] = reader
                shardPaths[shardFilename] = pathOf(shardFilename) ?: source.filePath

                // Add tensors with shard info
                val shardInfo = SafeTensorsIndexParser.parseShardFilename(shardFilename)
                for (tensorInfo in reader.tensors) {
                    _tensors.add(
                        ShardedTensorInfo(
                            base = tensorInfo,
                            shardFilename = shardFilename,
                            shardIndex = shardInfo?.first ?: (shardIndex + 1),
                            totalShards = shardInfo?.second ?: availableShards.size
                        )
                    )
                }

                // Merge metadata
                _metadata.putAll(reader.metadata)

                _loadedShards.add(shardFilename)

                _progress.emit(LoadingProgress.StageProgress(
                    stage,
                    1.0f,
                    itemsProcessed = reader.tensors.size,
                    totalItems = reader.tensors.size,
                    message = "Loaded ${reader.tensors.size} tensors"
                ))
                _progress.emit(LoadingProgress.StageCompleted(stage, currentTimeMillis() - shardStart))

            } catch (e: Exception) {
                source.close()
                if (!allowPartial) {
                    // Release what was opened: this reader is never returned to the caller.
                    close()
                    availableShards.drop(shardIndex + 1).forEach { openedShards.getValue(it).close() }
                    _progress.emit(LoadingProgress.Failed(stage, e, "Failed to load shard: $shardFilename"))
                    throw e
                }
                _missingShards.add(shardFilename)
            }
        }

        // Stage: aggregate
        val aggregateStage = LoadingStage("aggregate", "Aggregating", totalStages - 1, totalStages)
        _progress.emit(LoadingProgress.StageStarted(aggregateStage, "Combining tensor metadata"))
        val aggregateStart = currentTimeMillis()

        // Sort tensors by name for consistent ordering
        _tensors.sortBy { it.name }

        _progress.emit(LoadingProgress.StageCompleted(aggregateStage, currentTimeMillis() - aggregateStart))

        // Completed
        val totalDuration = currentTimeMillis() - startTime
        val summary = "Loaded ${_tensors.size} tensors from ${_loadedShards.size}/${shardFiles.size} shards"
        _progress.emit(LoadingProgress.Completed(totalDuration, summary))
    }

    // ========== Companion ==========

    public companion object {
        /**
         * Open a sharded SafeTensors model from its index file.
         *
         * @param indexPath Path to model.safetensors.index.json
         * @param allowPartial If true, continue loading even if some shards are missing
         * @return Reader with aggregated tensor metadata
         * @throws SafeTensorsIndexParseException if index is invalid
         * @throws SafeTensorsShardException if shards cannot be read
         */
        public suspend fun openFromIndex(
            indexPath: String,
            allowPartial: Boolean = false
        ): StreamingShardedSafeTensorsReader {
            // Read and parse index file
            val indexContent = readTextFile(indexPath)
                ?: throw SafeTensorsShardException.IndexNotFound(indexPath)

            val index = SafeTensorsIndexParser.parse(indexContent)

            // Shards live beside the index file
            val basePath = indexPath.substringBeforeLast("/")
            val resolve = { shardFilename: String -> resolveShardPath(basePath, shardFilename) }

            return open(index, indexPath, allowPartial, openShard = { openRandomAccessSource(resolve(it)) }, pathOf = resolve)
        }

        /**
         * Open a sharded SafeTensors model whose index is already in hand as text and whose shards
         * the caller provides — the entry point for hosts without a file system (#1246 follow-up).
         *
         * [shardSource] is asked once per shard name in the index (`model-00001-of-00003.safetensors`,
         * …) and answers with an open [RandomAccessSource] the reader owns from then on, or `null` for
         * a shard that is not there. It may suspend, so a browser can open a `Blob` per shard inside
         * it. Mapped loading ([loadTensorStorageMapped]) is available only for sources that carry a
         * [RandomAccessSource.filePath].
         *
         * @param indexJson The content of `model.safetensors.index.json`
         * @param shardSource Opens a shard by file name; `null` when it is missing
         * @param allowPartial If true, continue loading even if some shards are missing
         * @throws SafeTensorsIndexParseException if the index is invalid
         * @throws SafeTensorsShardException if shards cannot be read
         */
        public suspend fun openFromIndexContent(
            indexJson: String,
            shardSource: suspend (shardFilename: String) -> RandomAccessSource?,
            allowPartial: Boolean = false,
        ): StreamingShardedSafeTensorsReader =
            openFromParsedIndex(SafeTensorsIndexParser.parse(indexJson), shardSource, allowPartial)

        /** [openFromIndexContent] for an index the caller has parsed already. */
        public suspend fun openFromParsedIndex(
            index: SafeTensorsIndex,
            shardSource: suspend (shardFilename: String) -> RandomAccessSource?,
            allowPartial: Boolean = false,
        ): StreamingShardedSafeTensorsReader =
            open(index, "<index>", allowPartial, openShard = shardSource, pathOf = { null })

        private suspend fun open(
            index: SafeTensorsIndex,
            indexLabel: String,
            allowPartial: Boolean,
            openShard: suspend (String) -> RandomAccessSource?,
            pathOf: (String) -> String?,
        ): StreamingShardedSafeTensorsReader {
            val reader = StreamingShardedSafeTensorsReader(index, openShard, pathOf)

            // Emit parse_index stage (already done, but emit for consistency)
            val parseStage = LoadingStage("parse_index", "Parsing index", 0, 2 + index.shardCount + 1)
            reader._progress.emit(LoadingProgress.StageStarted(parseStage, indexLabel))
            reader._progress.emit(LoadingProgress.StageCompleted(parseStage, 0))

            // Load shards
            reader.loadShards(allowPartial)

            return reader
        }

        private fun resolveShardPath(basePath: String, shardFilename: String): String =
            if (basePath.endsWith("/")) basePath + shardFilename else "$basePath/$shardFilename"

        /**
         * Open a sharded SafeTensors model from any shard file.
         *
         * Automatically discovers the index file in the same directory.
         *
         * @param shardPath Path to any shard file (e.g., model-00001-of-00003.safetensors)
         * @param allowPartial If true, continue loading even if some shards are missing
         * @return Reader with aggregated tensor metadata
         * @throws SafeTensorsShardException if index not found or shards cannot be read
         */
        public suspend fun openFromShard(
            shardPath: String,
            allowPartial: Boolean = false
        ): StreamingShardedSafeTensorsReader {
            val filename = shardPath.substringAfterLast("/")
            val indexFilename = SafeTensorsIndexParser.deriveIndexFilename(filename)
                ?: throw SafeTensorsShardException.IndexNotFound(shardPath)

            val basePath = shardPath.substringBeforeLast("/")
            val indexPath = if (basePath.isEmpty()) indexFilename else "$basePath/$indexFilename"

            return openFromIndex(indexPath, allowPartial)
        }

        /**
         * Detect if a path is part of a sharded SafeTensors model.
         *
         * @param path Path to check (can be index file or shard file)
         * @return Pair of (isSharded, indexPath) or (false, null) if not sharded
         */
        public fun detectSharded(path: String): Pair<Boolean, String?> {
            // Check if it's an index file
            if (SafeTensorsIndexParser.isIndexFile(path)) {
                return true to path
            }

            // Check if it's a sharded filename
            val filename = path.substringAfterLast("/")
            if (SafeTensorsIndexParser.isShardedFilename(filename)) {
                val indexFilename = SafeTensorsIndexParser.deriveIndexFilename(filename)
                if (indexFilename != null) {
                    val basePath = path.substringBeforeLast("/")
                    val indexPath = if (basePath.isEmpty()) indexFilename else "$basePath/$indexFilename"
                    return true to indexPath
                }
            }

            return false to null
        }
    }
}

/**
 * Tensor info with shard location information.
 *
 * Wraps [StreamingSafeTensorInfo] and adds information about which shard
 * contains this tensor.
 */
public data class ShardedTensorInfo(
    /** The underlying tensor info from the shard reader */
    private val base: StreamingSafeTensorInfo,
    /** Filename of the shard containing this tensor */
    val shardFilename: String,
    /** 1-based shard index */
    val shardIndex: Int,
    /** Total number of shards */
    val totalShards: Int
) {
    // Delegate all properties from base
    val name: String get() = base.name
    val dtype: String get() = base.dtype
    val dataType get() = base.dataType
    val shape: List<Long> get() = base.shape
    val elementCount: Long get() = base.elementCount
    val dataOffsetStart: Long get() = base.dataOffsetStart
    val dataOffsetEnd: Long get() = base.dataOffsetEnd
    val sizeInBytes: Long get() = base.sizeInBytes
    val absoluteDataOffset: Long get() = base.absoluteDataOffset
    val isUnknownType: Boolean get() = base.isUnknownType

    /** Formatted shard location, e.g., "1/3" */
    val shardLocation: String get() = "$shardIndex/$totalShards"
}

/**
 * Exceptions specific to sharded SafeTensors operations.
 */
public sealed class SafeTensorsShardException(
    message: String,
    cause: Throwable? = null
) : Exception(message, cause) {

    /** Index file not found */
    public class IndexNotFound(
        val searchPath: String
    ) : SafeTensorsShardException("No index.json found for sharded model at $searchPath")

    /** Required shard file not found */
    public class ShardNotFound(
        val shardName: String,
        val expectedPath: String
    ) : SafeTensorsShardException("Shard not found: $shardName at $expectedPath")

    /** Some shards missing but partial load not allowed */
    public class IncompleteShard(
        val loadedCount: Int,
        val totalCount: Int,
        val missingShards: List<String>
    ) : SafeTensorsShardException(
        "Incomplete model: loaded $loadedCount/$totalCount shards, missing: ${missingShards.joinToString()}"
    )
}

// ========== Platform Utilities ==========

/**
 * Read a text file. Platform-specific implementation.
 */
public expect fun readTextFile(path: String): String?

/**
 * Get current time in milliseconds.
 */
internal expect fun currentTimeMillis(): Long
