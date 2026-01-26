package sk.ainet.io.model

import kotlinx.coroutines.flow.Flow

/**
 * Represents a stage in the loading process.
 *
 * @property id Unique stage identifier (e.g., "parse_index", "load_shard_1")
 * @property name Human-readable name for display
 * @property order Stage order (0-based)
 * @property totalStages Total number of stages in the operation
 */
public data class LoadingStage(
    val id: String,
    val name: String,
    val order: Int,
    val totalStages: Int
)

/**
 * Represents progress within a loading operation.
 *
 * This sealed class hierarchy provides type-safe progress events that can be
 * collected via Kotlin Flow. Designed for use with coroutines - no 3rd party
 * reactive libraries required.
 *
 * Usage:
 * ```kotlin
 * loader.progress.collect { progress ->
 *     when (progress) {
 *         is LoadingProgress.Started -> println("Loading started")
 *         is LoadingProgress.StageStarted -> println("Stage: ${progress.stage.name}")
 *         is LoadingProgress.StageProgress -> updateProgressBar(progress.progress)
 *         is LoadingProgress.StageCompleted -> println("Done: ${progress.stage.name}")
 *         is LoadingProgress.Completed -> println("All done!")
 *         is LoadingProgress.Failed -> println("Error: ${progress.message}")
 *     }
 * }
 * ```
 */
public sealed class LoadingProgress {

    /**
     * Loading has started.
     *
     * @property totalStages Total number of stages to be executed
     * @property description Optional description of the overall operation
     */
    public data class Started(
        val totalStages: Int,
        val description: String? = null
    ) : LoadingProgress()

    /**
     * A new stage has begun.
     *
     * @property stage The stage that is starting
     * @property description Optional description of what this stage does
     */
    public data class StageStarted(
        val stage: LoadingStage,
        val description: String? = null
    ) : LoadingProgress()

    /**
     * Progress within the current stage.
     *
     * @property stage The current stage
     * @property progress Progress value from 0.0 to 1.0
     * @property bytesProcessed Optional bytes processed so far
     * @property totalBytes Optional total bytes to process
     * @property itemsProcessed Optional items processed so far
     * @property totalItems Optional total items to process
     * @property message Optional status message
     */
    public data class StageProgress(
        val stage: LoadingStage,
        val progress: Float,
        val bytesProcessed: Long? = null,
        val totalBytes: Long? = null,
        val itemsProcessed: Int? = null,
        val totalItems: Int? = null,
        val message: String? = null
    ) : LoadingProgress()

    /**
     * A stage has completed.
     *
     * @property stage The completed stage
     * @property durationMs Time taken for this stage in milliseconds
     */
    public data class StageCompleted(
        val stage: LoadingStage,
        val durationMs: Long
    ) : LoadingProgress()

    /**
     * Overall loading completed successfully.
     *
     * @property totalDurationMs Total time taken in milliseconds
     * @property summary Optional summary of the operation
     */
    public data class Completed(
        val totalDurationMs: Long,
        val summary: String? = null
    ) : LoadingProgress()

    /**
     * Loading failed with an error.
     *
     * @property stage Stage where failure occurred, null if before any stage
     * @property error The exception that caused the failure
     * @property message Human-readable error message
     */
    public data class Failed(
        val stage: LoadingStage?,
        val error: Throwable,
        val message: String
    ) : LoadingProgress()
}

/**
 * Interface for loaders that support progress reporting.
 *
 * Consuming applications can collect the [progress] flow to receive updates
 * during loading operations. The flow uses SharedFlow internally, supporting
 * multiple collectors.
 *
 * This interface is optional - loaders that don't need progress reporting
 * don't need to implement it.
 *
 * Usage:
 * ```kotlin
 * val loader: ProgressReportingLoader = ...
 *
 * // Start collecting before loading
 * val progressJob = scope.launch {
 *     loader.progress.collect { progress ->
 *         updateUI(progress)
 *     }
 * }
 *
 * // Perform load operation
 * val result = loader.load(...)
 *
 * // Cancel collection when done
 * progressJob.cancel()
 * ```
 */
public interface ProgressReportingLoader {

    /**
     * Flow of loading progress events.
     *
     * Collect this flow before calling load operations to receive progress updates.
     * Uses SharedFlow internally - multiple collectors are supported.
     */
    public val progress: Flow<LoadingProgress>
}
