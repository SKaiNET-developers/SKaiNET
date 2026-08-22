@file:JvmName("DTypeBridge")
@file:Suppress("DEPRECATION") // the bridge exists to keep the deprecated type usable until removal

package sk.ainet.lang.tensor.storage

import sk.ainet.lang.types.DType
import kotlin.jvm.JvmName

/**
 * The [LogicalDType] describing this [DType] — bridge half 2 of 2 (SKEEP-003 Phase 0,
 * decision #13).
 *
 * Lives in the storage package (not in `sk.ainet.lang.types`) so the type package stays a
 * leaf; the mapping is total and bijective with [LogicalDType.toDType]:
 *
 * ```
 * for (l in LogicalDType.entries) check(l.toDType().toLogicalDType() == l)
 * ```
 *
 * When `LogicalDType` is deprecated (decision #13, separate slice) its `ReplaceWith` targets
 * point at `DType` directly and this extension becomes a no-op shim.
 */
@Deprecated(
    message = "LogicalDType merges into DType (SKEEP-003 decision #13); keep the DType itself.",
    replaceWith = ReplaceWith("this"),
)
public fun DType.toLogicalDType(): LogicalDType = LogicalDType.fromDType(this)
