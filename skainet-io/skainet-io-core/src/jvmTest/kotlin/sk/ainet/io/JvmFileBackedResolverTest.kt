package sk.ainet.io

import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.tensor.storage.ByteArrayAccessor
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JvmFileBackedResolverTest {

    private fun withTempFile(content: ByteArray, block: (File) -> Unit) {
        val file = File.createTempFile("resolver_test_", ".bin")
        try {
            file.writeBytes(content)
            block(file)
        } finally {
            file.delete()
        }
    }

    @Test
    fun resolveFileBackedHandleReadsFull() {
        val data = ByteArray(256) { it.toByte() }
        withTempFile(data) { file ->
            val handle = BufferHandle.FileBacked(
                path = file.absolutePath,
                fileOffset = 0,
                sizeInBytes = 256
            )
            val resolver = JvmFileBackedResolver.createResolver()
            val accessor = resolver.resolve(handle)

            assertEquals(256L, accessor.sizeInBytes)
            assertEquals(0.toByte(), accessor.readByte(0))
            assertEquals(255.toByte(), accessor.readByte(255))

            val bytes = accessor.readBytes(10, 5)
            assertEquals(5, bytes.size)
            assertEquals(10.toByte(), bytes[0])
            assertEquals(14.toByte(), bytes[4])

            accessor.close()
        }
    }

    @Test
    fun resolveFileBackedHandleWithOffset() {
        val data = ByteArray(1024) { it.toByte() }
        withTempFile(data) { file ->
            val handle = BufferHandle.FileBacked(
                path = file.absolutePath,
                fileOffset = 512,
                sizeInBytes = 100
            )
            val resolver = JvmFileBackedResolver.createResolver()
            val accessor = resolver.resolve(handle)

            assertEquals(100L, accessor.sizeInBytes)
            // First byte of the mapped region should be byte 512 of the file
            assertEquals(0.toByte(), accessor.readByte(0)) // 512 % 256 = 0
            accessor.close()
        }
    }

    @Test
    fun resolveOwnedHandleDirectly() {
        val data = byteArrayOf(10, 20, 30, 40)
        val handle = BufferHandle.Owned(data)
        val resolver = JvmFileBackedResolver.createResolver()
        val accessor = resolver.resolve(handle)

        assertTrue(accessor is ByteArrayAccessor)
        assertEquals(4L, accessor.sizeInBytes)
        assertEquals(10.toByte(), accessor.readByte(0))
        accessor.close()
    }

    @Test
    fun resolveBorrowedHandleDirectly() {
        val data = byteArrayOf(5, 6, 7)
        val handle = BufferHandle.Borrowed(data)
        val resolver = JvmFileBackedResolver.createResolver()
        val accessor = resolver.resolve(handle)

        assertTrue(accessor is ByteArrayAccessor)
        assertEquals(3L, accessor.sizeInBytes)
        accessor.close()
    }

    @Test
    fun resolveAliasedHandle() {
        val data = ByteArray(100) { it.toByte() }
        val parent = BufferHandle.Owned(data)
        val alias = BufferHandle.Aliased(parent, byteOffset = 10, sizeInBytes = 20)
        val resolver = JvmFileBackedResolver.createResolver()
        val accessor = resolver.resolve(alias)

        assertEquals(20L, accessor.sizeInBytes)
        assertEquals(10.toByte(), accessor.readByte(0))
        assertEquals(29.toByte(), accessor.readByte(19))
        accessor.close()
    }

    @Test
    fun readAllBytesFromFileBacked() {
        val data = byteArrayOf(1, 2, 3, 4, 5)
        withTempFile(data) { file ->
            val handle = BufferHandle.FileBacked(file.absolutePath, 0, 5)
            val resolver = JvmFileBackedResolver.createResolver()
            val accessor = resolver.resolve(handle)

            val all = accessor.readAllBytes()
            assertEquals(5, all.size)
            assertEquals(1.toByte(), all[0])
            assertEquals(5.toByte(), all[4])
            accessor.close()
        }
    }
}
