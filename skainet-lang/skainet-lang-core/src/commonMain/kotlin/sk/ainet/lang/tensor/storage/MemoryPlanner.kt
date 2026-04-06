package sk.ainet.lang.tensor.storage

/**
 * Resolves [Placement] intent into concrete buffer allocation decisions.
 *
 * The planner inspects available backends and decides:
 * - Where a tensor should actually live (device + memory domain)
 * - Whether a fallback is needed (e.g. GPU not available → CPU)
 * - Whether immutable weights should be file-backed vs heap-copied
 *
 * Currently only the CPU backend is wired in, so the planner always
 * resolves to CPU/HOST_HEAP or CPU/MMAP_FILE. GPU/NPU resolution
 * will be added when those backends ship.
 */
public class MemoryPlanner(
    private val availableDevices: Set<DeviceKind> = setOf(DeviceKind.CPU)
) {

    /**
     * Resolve a placement intent to an actual placement that can be satisfied.
     *
     * @param requested  The user/loader-requested placement
     * @return A [ResolvedPlacement] with the actual target and whether fallback was used
     */
    public fun resolve(requested: Placement): ResolvedPlacement {
        val targetDevice = if (requested.device == DeviceKind.AUTO) {
            bestAvailableDevice()
        } else {
            requested.device
        }

        return if (targetDevice in availableDevices) {
            ResolvedPlacement(
                actual = requested.copy(device = targetDevice),
                usedFallback = false
            )
        } else if (requested.requirement == Requirement.REQUIRED) {
            throw PlacementUnavailableException(
                "Required device $targetDevice is not available. Available: $availableDevices"
            )
        } else {
            // Fallback to the placement's specified fallback device
            val fallbackDevice = if (requested.fallback in availableDevices) {
                requested.fallback
            } else {
                DeviceKind.CPU
            }
            ResolvedPlacement(
                actual = Placement(
                    device = fallbackDevice,
                    domain = fallbackDomain(requested.domain, fallbackDevice),
                    residency = requested.residency,
                    requirement = requested.requirement,
                    fallback = requested.fallback
                ),
                usedFallback = true
            )
        }
    }

    /**
     * Suggest the best placement for a weight tensor.
     * File-backed if persistent, heap if transient.
     */
    public fun suggestWeightPlacement(isFileBacked: Boolean): Placement {
        return if (isFileBacked) Placement.MMAP_WEIGHTS else Placement.CPU_HEAP.copy(residency = Residency.PERSISTENT)
    }

    /**
     * Suggest placement for a mutable activation/intermediate tensor.
     */
    public fun suggestActivationPlacement(): Placement = Placement.CPU_HEAP

    private fun bestAvailableDevice(): DeviceKind = when {
        DeviceKind.GPU in availableDevices -> DeviceKind.GPU
        DeviceKind.NPU in availableDevices -> DeviceKind.NPU
        else -> DeviceKind.CPU
    }

    private fun fallbackDomain(requested: MemoryDomain, device: DeviceKind): MemoryDomain {
        // If falling back to CPU, translate device-specific domains to host domains
        return when {
            device == DeviceKind.CPU && requested == MemoryDomain.DEVICE_LOCAL -> MemoryDomain.HOST_HEAP
            device == DeviceKind.CPU && requested == MemoryDomain.UNIFIED -> MemoryDomain.HOST_HEAP
            else -> requested
        }
    }
}

public data class ResolvedPlacement(
    val actual: Placement,
    val usedFallback: Boolean
)

public class PlacementUnavailableException(message: String) : RuntimeException(message)
