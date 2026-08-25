package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.blockSpec
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * Decides the [WeightForm] a weight should take, from what the file holds, what the device is, and
 * what the backend's kernels can feed (#1109).
 *
 * This is the piece that lets a model author declare nothing. The same `nn { }` graph runs over an
 * FP32 checkpoint on a workstation and a Q4_K GGUF on a phone, and the difference in how the
 * weights should be held is a function of three things the author does not know and the loader
 * does.
 */
@ExperimentalMemoryApi
public object WeightFormResolver {

    /**
     * The form a weight stored as [stored] should take on a device described by [profile], given
     * what [capabilities] can feed.
     *
     * The rules, in order:
     *
     * The shape axis ([WeightShapeOrientation]) is deliberately not resolved here. Which way round
     * a weight's dimensions are labelled is a property of the *checkpoint convention*, not of the
     * device, and reversing it changes every consumer's idea of a tensor's shape — so it stays the
     * caller's explicit choice, defaulting to what the file says.
     *
     * 1. **Residency** comes from the profile alone. `weightsMapped` is a statement about the
     *    device — a 2 GB board cannot hold the weights on the heap whatever they are encoded as.
     * 2. **A dense weight** is already in the only form it has.
     * 3. **A kernel can feed the stored encoding** → keep it. Ask only whether the kernel wants
     *    its blocks in feed order, which every packed kernel in the tree does, and hand it bytes
     *    that way so the per-weight relayout (#1096) never has to run.
     * 4. **Nothing can feed it** → the backend would otherwise dequantize on *every forward pass*.
     *    Doing it once at load is strictly better, so that is the default; but it is also a real
     *    cost — FP32 is roughly eight times a Q4_K tensor — and a profile that says [PlannerProfile.strict]
     *    means a missing kernel is a bug to surface, not a slow path to take quietly.
     *
     * @throws IllegalStateException when nothing can feed [stored] and the profile is strict
     */
    public fun resolve(
        stored: TensorEncoding?,
        profile: PlannerProfile,
        capabilities: KernelCapabilities,
    ): WeightForm {
        val residency = if (profile.weightsMapped) WeightResidency.MAPPED else WeightResidency.HEAP

        if (stored == null || stored is TensorEncoding.Dense) {
            return WeightForm(EncodingRequest.KeepAsStored, WeightByteOrder.AS_STORED, residency = residency)
        }

        if (capabilities.canFeedMatmul(stored)) {
            val order =
                if (capabilities.wantsKernelFeedOrder(stored)) WeightByteOrder.KERNEL_FEED
                else WeightByteOrder.AS_STORED
            return WeightForm(EncodingRequest.KeepAsStored, order, residency = residency)
        }

        check(!profile.strict) {
            "no kernel on this target can feed a ${stored.name} weight, and ${profile.name} is strict: " +
                "loading it would dequantize to FP32 — about ${dequantizedTimes(stored)}× the bytes — and the " +
                "profile asks to be told rather than to pay that quietly. Register a ${stored.name} kernel, " +
                "convert the file, or resolve with a non-strict profile."
        }

        // Once, at load, instead of once per forward pass.
        return WeightForm(EncodingRequest.DequantizeTo(FP32), WeightByteOrder.AS_STORED, residency = residency)
    }

    /** Roughly how much bigger [encoding] gets as dense FP32 — for the message, not for the plan. */
    private fun dequantizedTimes(encoding: TensorEncoding): String {
        val bits = encoding.blockSpec?.bitsPerElement ?: return "several"
        val tenths = ((32.0 / bits) * 10).toInt()
        return "${tenths / 10}.${tenths % 10}"
    }
}
