package sk.ainet.lang.nn.optim

import kotlin.math.PI
import kotlin.math.cos

/**
 * A learning-rate schedule: maps a global training step to a learning rate.
 *
 * Schedules are stateless and decoupled from optimizers — a training loop
 * assigns the value to the optimizer's mutable `lr` before each step:
 *
 * ```kotlin
 * val schedule = linearWarmupCosineDecay(totalSteps, warmupSteps, peakLr = 5e-4)
 * // inside the loop:
 * optimizer.lr = schedule.lrAt(globalStep)
 * ```
 */
public fun interface LrSchedule {
    /** The learning rate to use for [step] (0-based, `0 <= step < totalSteps`). */
    public fun lrAt(step: Int): Double
}

/**
 * Linear warmup followed by cosine decay — the schedule used by GPT-style
 * pretraining recipes.
 *
 * - steps `0 ..< warmupSteps`: linear ramp from [initialLr] to [peakLr]
 * - steps `warmupSteps ..< totalSteps`: cosine decay from [peakLr] to [minLr]
 *
 * @param totalSteps total number of optimization steps
 * @param warmupSteps steps spent ramping up (must be in `1 ..< totalSteps`)
 * @param peakLr learning rate at the end of the warmup
 * @param initialLr learning rate at step 0
 * @param minLr learning rate approached at the end of training
 */
public fun linearWarmupCosineDecay(
    totalSteps: Int,
    warmupSteps: Int,
    peakLr: Double,
    initialLr: Double = 3e-5,
    minLr: Double = 1e-6,
): LrSchedule {
    require(totalSteps > 0) { "totalSteps must be positive, was $totalSteps" }
    require(warmupSteps in 1 until totalSteps) {
        "warmupSteps must be in 1 until totalSteps=$totalSteps, was $warmupSteps"
    }
    return LrSchedule { step ->
        require(step in 0 until totalSteps) { "step $step out of range [0, $totalSteps)" }
        if (step < warmupSteps) {
            initialLr + (peakLr - initialLr) * step / warmupSteps
        } else {
            val progress = (step - warmupSteps).toDouble() / (totalSteps - warmupSteps)
            minLr + (peakLr - minLr) * 0.5 * (1 + cos(PI * progress))
        }
    }
}
