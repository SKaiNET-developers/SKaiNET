package sk.ainet.io.irpa

import kotlinx.io.Sink
import kotlinx.io.write
import kotlinx.io.writeIntLe
import kotlinx.io.writeLongLe
import kotlinx.io.writeShortLe
import sk.ainet.compile.hlo.ExternalParameterRef
import sk.ainet.lang.tensor.storage.BufferHandle

/**
 * Writes an IREE parameter archive (`.irpa`) file.
 *
 * The archive is the runtime-consumable counterpart to the
 * `#flow.parameter.named<"scope"::"key">` references emitted by
 * [sk.ainet.compile.hlo.StableHloConverter] when
 * [sk.ainet.compile.hlo.ConstantMaterializationPolicy.ExternalAlways] is
 * active. `iree-compile --iree-opt-import-parameters=<path>.irpa` inlines
 * the referenced bytes at compile time; `iree-run-module --parameters=<scope>=<path>.irpa`
 * loads them at runtime.
 *
 * Format: v0 binary layout defined in IREE's `parameter_archive.h`:
 *
 * ```
 *   +-------------------------- 40 B --------------------------+
 *   | header_v0 (magic="IRPA", segments)                        |
 *   +-------------------------- pad to 16 ----------------------+
 *   | entry_segment (DATA entries, each 80 B aligned to 16)     |
 *   +-------------------------- no padding ---------------------+
 *   | metadata_segment (concatenated key bytes, per entry)      |
 *   +-------------------------- pad to entry.minimum_alignment -+
 *   | storage_segment (tensor bytes, each block aligned to 64)  |
 *   +-----------------------------------------------------------+
 * ```
 *
 * All `u16`/`u32`/`u64` values little-endian. Offsets inside the header
 * are relative to the header start; inside entries, to the owning
 * segment's offset. The archive has no dtype, no shape, and no scope
 * column — the MLIR `#flow.parameter.named<...>` reference carries all
 * structural metadata, and scope binding is a CLI concern.
 *
 * This writer is scope-agnostic: a single [write] call produces one
 * `.irpa` for all entries passed in. Callers with tensors in multiple
 * scopes must call [write] once per scope (use [groupByScope]).
 *
 * See issue #523 for the architectural context.
 */
public class IrpaWriter {

    /**
     * Write an `.irpa` archive containing every [entry] to [sink].
     *
     * Streams the output so peak memory stays bounded by the largest
     * single [BufferHandle] and the entry/metadata tables (tiny even
     * for thousands of weights). The sink is not flushed or closed —
     * caller decides when to do either.
     *
     * @throws IllegalArgumentException if [entries] is empty — a
     *     valid archive must have at least one entry, and silently
     *     emitting an empty file is a worse failure mode than a loud
     *     precondition check.
     */
    public fun write(entries: List<ExternalParameterRef>, sink: Sink) {
        require(entries.isNotEmpty()) {
            "IrpaWriter.write() requires at least one entry; empty archives are not useful."
        }

        // Precompute layout. Segment offsets are relative to the
        // header start (first byte of the file, since we always write
        // the header at offset 0).
        val headerBlockSize = HEADER_BLOCK_SIZE
        val entrySegmentOffset = alignUp(headerBlockSize, HEADER_ALIGNMENT).toLong()
        val entrySegmentLength = entries.size * ENTRY_SIZE_ALIGNED.toLong()
        val metadataSegmentOffset = entrySegmentOffset + entrySegmentLength
        val metadataSegmentLength = entries.sumOf { it.key.encodeToByteArray().size.toLong() }

        // Storage segment starts at the first alignment >= end-of-metadata
        // that satisfies DEFAULT_DATA_ALIGNMENT. Per-entry alignment
        // inside the segment uses the same default unless an entry
        // overrides it (not exposed today).
        val storageSegmentOffset = alignUpLong(
            metadataSegmentOffset + metadataSegmentLength,
            DEFAULT_DATA_ALIGNMENT.toLong()
        )

        // Running storage cursor — offset WITHIN the storage segment,
        // not absolute. Each entry records its offset here after
        // aligning up.
        var storageCursor = 0L
        val perEntryStorageOffset = LongArray(entries.size)
        for ((i, entry) in entries.withIndex()) {
            val aligned = alignUpLong(storageCursor, DEFAULT_DATA_ALIGNMENT.toLong())
            perEntryStorageOffset[i] = aligned
            storageCursor = aligned + entry.source.sizeInBytes
        }
        val storageSegmentLength = storageCursor

        // --- Write the header ---
        writeHeader(
            sink = sink,
            entryCount = entries.size.toLong(),
            entrySegmentOffset = entrySegmentOffset,
            entrySegmentLength = entrySegmentLength,
            metadataSegmentOffset = metadataSegmentOffset,
            metadataSegmentLength = metadataSegmentLength,
            storageSegmentOffset = storageSegmentOffset,
            storageSegmentLength = storageSegmentLength
        )

        // Pad from end-of-header-block to start-of-entry-segment.
        writePadding(sink, (entrySegmentOffset - headerBlockSize.toLong()).toInt())

        // --- Write the entry segment (DATA records) ---
        var metadataCursor = 0L
        for ((i, entry) in entries.withIndex()) {
            val keyBytes = entry.key.encodeToByteArray()
            writeDataEntry(
                sink = sink,
                nameOffset = metadataCursor,
                nameLength = keyBytes.size.toLong(),
                storageOffset = perEntryStorageOffset[i],
                storageLength = entry.source.sizeInBytes
            )
            metadataCursor += keyBytes.size
        }

        // --- Write the metadata segment (concatenated key bytes) ---
        for (entry in entries) {
            val keyBytes = entry.key.encodeToByteArray()
            for (b in keyBytes) sink.writeByte(b)
        }

        // Pad from end-of-metadata to start-of-storage.
        val metadataEndAbs = metadataSegmentOffset + metadataSegmentLength
        writePaddingLong(sink, storageSegmentOffset - metadataEndAbs)

        // --- Write the storage segment ---
        var writtenInStorage = 0L
        for ((i, entry) in entries.withIndex()) {
            // Pad from previous entry's end to this entry's aligned offset.
            val entryOffset = perEntryStorageOffset[i]
            writePaddingLong(sink, entryOffset - writtenInStorage)
            writeBufferHandle(sink, entry.source)
            writtenInStorage = entryOffset + entry.source.sizeInBytes
        }
        // No trailing file-level pad to 4096 — optional per the spec,
        // and callers writing to a stream may not know total size up
        // front. mmap readers tolerate short tails.
    }

    /**
     * Group a mixed-scope ref list into per-scope bundles. Callers
     * with multiple scopes should invoke [write] once per bundle and
     * pass each resulting file to `iree-compile --parameters=<scope>=<path>`.
     *
     * Preserves within-scope order — matters for reproducible archives.
     */
    public fun groupByScope(entries: List<ExternalParameterRef>): Map<String, List<ExternalParameterRef>> {
        val grouped = linkedMapOf<String, MutableList<ExternalParameterRef>>()
        for (entry in entries) {
            grouped.getOrPut(entry.scope) { mutableListOf() }.add(entry)
        }
        return grouped
    }

    private fun writeHeader(
        sink: Sink,
        entryCount: Long,
        entrySegmentOffset: Long,
        entrySegmentLength: Long,
        metadataSegmentOffset: Long,
        metadataSegmentLength: Long,
        storageSegmentOffset: Long,
        storageSegmentLength: Long
    ) {
        // --- Fixed 40-byte header (iree_io_parameter_archive_header_v0_t) ---
        sink.writeIntLe(MAGIC)                         //  0: magic
        sink.writeShortLe(0)                           //  4: version_major
        sink.writeShortLe(0)                           //  6: version_minor
        sink.writeLongLe(HEADER_FIXED_SIZE.toLong())   //  8: header_size (40)
        sink.writeLongLe(0L)                           // 16: next_header_offset
        sink.writeLongLe(0L)                           // 24: flags
        sink.writeLongLe(entryCount)                   // 32: entry_count

        // --- Three segment references, 16 bytes each ---
        // Layout: { u64 offset; u64 length; } — offsets are relative
        // to the start of the header block.
        sink.writeLongLe(entrySegmentOffset)           // 40: entry.offset
        sink.writeLongLe(entrySegmentLength)           // 48: entry.length
        sink.writeLongLe(metadataSegmentOffset)        // 56: metadata.offset
        sink.writeLongLe(metadataSegmentLength)        // 64: metadata.length
        sink.writeLongLe(storageSegmentOffset)         // 72: storage.offset
        sink.writeLongLe(storageSegmentLength)         // 80: storage.length
        // Total written: 88 bytes (HEADER_BLOCK_SIZE).
    }

    private fun writeDataEntry(
        sink: Sink,
        nameOffset: Long,
        nameLength: Long,
        storageOffset: Long,
        storageLength: Long
    ) {
        sink.writeLongLe(ENTRY_HEADER_SIZE_DATA.toLong())  // entry_size (u64)
        sink.writeIntLe(ENTRY_TYPE_DATA)                   // type (u32)
        // 4-byte pad: the C struct has `u64 flags` immediately after
        // `u32 type`, and the compiler inserts 4 bytes of padding to
        // align `flags` on an 8-byte boundary. Without these bytes
        // every subsequent u64 field reads from the wrong offset and
        // the parser rejects the archive.
        sink.writeIntLe(0)
        sink.writeLongLe(0L)                               // flags (u64)
        sink.writeLongLe(nameOffset)                       // name.offset
        sink.writeLongLe(nameLength)                       // name.length
        sink.writeLongLe(0L)                               // metadata.offset
        sink.writeLongLe(0L)                               // metadata.length
        sink.writeLongLe(DEFAULT_DATA_ALIGNMENT.toLong())  // minimum_alignment
        sink.writeLongLe(storageOffset)                    // storage.offset
        sink.writeLongLe(storageLength)                    // storage.length
        // Total bytes written: 80 (ENTRY_HEADER_SIZE_DATA). Already
        // 16-aligned, so no inter-entry padding required.
    }

    private fun writeBufferHandle(sink: Sink, handle: BufferHandle) {
        when (handle) {
            is BufferHandle.Owned -> writeByteArray(sink, handle.data, handle.offset, handle.sizeInBytes.toInt())
            is BufferHandle.Borrowed -> writeByteArray(sink, handle.data, handle.offset, handle.sizeInBytes.toInt())
            is BufferHandle.FileBacked -> writeFileBackedBytes(sink, handle)
            else -> throw IllegalArgumentException(
                "IrpaWriter does not yet handle BufferHandle subclass ${handle::class.simpleName}. " +
                    "Owned / Borrowed / FileBacked are wired. Aliased, DeviceResident, and " +
                    "other variants are out of scope — resolve them to one of the wired " +
                    "variants before handing to the writer."
            )
        }
    }

    private fun writeByteArray(sink: Sink, data: ByteArray, offset: Int, length: Int) {
        // Byte-at-a-time for the same reason noted below — and because
        // under the sizes we see in practice for single-op values
        // (tens to a few thousand bytes) the overhead is lost in the
        // wider write cost. FileBacked paths use a chunked copy on
        // their platform-specific side, which is where the byte
        // volume is meaningful.
        for (i in offset until offset + length) {
            sink.writeByte(data[i])
        }
    }

    private fun writePadding(sink: Sink, bytes: Int) {
        if (bytes <= 0) return
        val zeros = ByteArray(bytes)
        sink.write(zeros)
    }

    private fun writePaddingLong(sink: Sink, bytes: Long) {
        if (bytes <= 0) return
        // Write in chunks to keep the transient buffer small; 4 KiB
        // is comfortably below any real alignment gap we'll see.
        val chunk = ByteArray(4096)
        var remaining = bytes
        while (remaining > 0) {
            val step = if (remaining >= chunk.size) chunk.size else remaining.toInt()
            if (step == chunk.size) {
                sink.write(chunk)
            } else {
                sink.write(ByteArray(step))
            }
            remaining -= step
        }
    }

    public companion object {
        /** Magic bytes `"IRPA"` little-endian. */
        public const val MAGIC: Int = 0x41505249
        /**
         * Size of the fixed `iree_io_parameter_archive_header_v0_t`
         * struct — magic through entry_count. This is the value
         * written into the header's own `header_size` field.
         */
        public const val HEADER_FIXED_SIZE: Int = 40
        /**
         * Fixed size of the three segment references that follow the
         * header struct (3 × 16 bytes).
         */
        public const val SEGMENT_REFS_SIZE: Int = 48
        /**
         * Total on-disk size of the header + segment-references block
         * before the entry segment begins. Entry segment offset is
         * always >= this value.
         */
        public const val HEADER_BLOCK_SIZE: Int = HEADER_FIXED_SIZE + SEGMENT_REFS_SIZE
        /** Alignment the header itself sits at in the file. */
        public const val HEADER_ALIGNMENT: Int = 16
        /**
         * Natural size of a DATA-type entry header. Each entry is
         * then padded to [ENTRY_SIZE_ALIGNED] before the next entry
         * starts.
         */
        public const val ENTRY_HEADER_SIZE_DATA: Int = 80
        /** Entry-to-entry alignment inside the entry segment. */
        public const val ENTRY_ALIGNMENT: Int = 16
        /** DATA entry size after alignment padding. */
        public const val ENTRY_SIZE_ALIGNED: Int = ENTRY_HEADER_SIZE_DATA  // already 16-aligned
        /** `iree_io_parameter_archive_entry_type_t::DATA`. */
        public const val ENTRY_TYPE_DATA: Int = 2
        /** Default per-entry storage alignment. */
        public const val DEFAULT_DATA_ALIGNMENT: Int = 64

        /**
         * Round [value] up to the nearest multiple of [alignment].
         * Used for header / entry alignment (always `Int` inputs).
         */
        internal fun alignUp(value: Int, alignment: Int): Int =
            (value + alignment - 1) and (alignment - 1).inv()

        /**
         * Round [value] up to the nearest multiple of [alignment].
         * `Long` variant for storage-segment offsets, which can exceed
         * 2 GiB for large models.
         */
        internal fun alignUpLong(value: Long, alignment: Long): Long =
            (value + alignment - 1L) and (alignment - 1L).inv()
    }
}
