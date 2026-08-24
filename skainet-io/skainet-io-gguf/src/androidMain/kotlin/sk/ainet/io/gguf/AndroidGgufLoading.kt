package sk.ainet.io.gguf

import android.app.ActivityManager
import android.content.Context
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.openRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.io.model.StagingPolicy
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.Budget
import sk.ainet.lang.memory.plan.DeviceFit
import sk.ainet.lang.memory.plan.DeviceMemory
import sk.ainet.lang.memory.plan.MemoryPlan
import sk.ainet.lang.memory.plan.MemoryPlans
import sk.ainet.lang.memory.plan.fitOn

/**
 * Loading a GGUF on Android: mapped weights by default, and a fit check *before* the load
 * (SKEEP-002, #921, #922, #1038).
 *
 * The managed heap is the binding constraint on a phone — hard-capped at 256 MB (512 MB with
 * `largeHeap`) no matter how much RAM the device has — so the Android configuration of the loader
 * is `staging = MAPPED`: weights come from file-backed pages the OS pages in on demand and evicts
 * under pressure, and never count against the cap.
 *
 * What is *not* solved yet: packed (quantized) tensors still arrive as heap arrays, because the
 * packed kernels take `ByteArray`s until the view contract of #973 lands. Mapping therefore lifts
 * the ceiling for dense-F32 weights today, and for a Q4_K_M checkpoint only once #973 does. The
 * fit check tells you which of the two pools you are about to run out of, rather than letting the
 * app find out by being killed.
 */
@OptIn(ExperimentalMemoryApi::class)
public object AndroidGguf {

    /**
     * The loader Android should use: positional reads for the metadata, mapped pages for tensor
     * payloads. [quantPolicy] is the caller's choice as usual; [staging] defaults to
     * [StagingPolicy.MAPPED] and is a parameter only so a test or a benchmark can ask for the
     * heap path explicitly.
     */
    public fun loader(
        filePath: String,
        quantPolicy: QuantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
        staging: StagingPolicy = StagingPolicy.MAPPED,
        onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
    ): StreamingGgufParametersLoader = StreamingGgufParametersLoader(
        sourceProvider = { openSource(filePath) },
        onProgress = onProgress,
        quantPolicy = quantPolicy,
        staging = staging,
    )

    /**
     * What this device has to offer: `ActivityManager.MemoryInfo` for physical RAM plus the ART
     * heap cap, which is what actually stops a model from loading.
     */
    public fun deviceMemory(context: Context): DeviceMemory {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        val runtime = Runtime.getRuntime()
        return DeviceMemory(
            totalRamBytes = info.totalMem,
            availableRamBytes = info.availMem,
            heapMaxBytes = runtime.maxMemory(),
            heapUsedBytes = runtime.totalMemory() - runtime.freeMemory(),
            lowMemory = info.lowMemory,
            lowMemoryThresholdBytes = info.threshold,
        )
    }

    /**
     * The plan a GGUF's *header* predicts at [ctx] — tensor table and metadata only, no payload
     * (M0-F1), so this costs a few kilobytes and a couple of reads.
     */
    public fun plan(filePath: String, ctx: Int, budget: Budget? = null): MemoryPlan =
        openSource(filePath).use { source ->
            MemoryPlans.plan(StreamingGGUFReader.open(source).planInput(ctx), budget)
        }

    private fun openSource(filePath: String): RandomAccessSource =
        openRandomAccessSource(filePath)
            ?: throw IllegalArgumentException("Cannot open for random access: $filePath")

    /**
     * Will this model load on this device? Checks the header-derived plan against both pools —
     * managed heap and physical RAM — before a byte of payload is read.
     *
     * @param weightsMapped whether the load will use [StagingPolicy.MAPPED] (what [loader] does)
     */
    public fun fits(context: Context, filePath: String, ctx: Int, weightsMapped: Boolean = true): DeviceFit =
        fits(deviceMemory(context), filePath, ctx, weightsMapped)

    /** [fits] against an explicit [DeviceMemory] — the form a test or a simulation uses. */
    public fun fits(device: DeviceMemory, filePath: String, ctx: Int, weightsMapped: Boolean = true): DeviceFit =
        plan(filePath, ctx).fitOn(device, weightsMapped)
}
