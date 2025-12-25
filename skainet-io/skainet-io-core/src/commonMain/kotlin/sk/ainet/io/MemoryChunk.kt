package sk.ainet.io

interface MemoryChunk {
    val size: Long
    
    fun readByte(offset: Long): Byte
    fun readBytes(offset: Long, length: Int): ByteArray
    
    fun slice(offset: Long, length: Long): MemoryChunk
}

class ByteArrayMemoryChunk(
    private val data: ByteArray,
    private val offset: Int = 0,
    override val size: Long = data.size.toLong()
) : MemoryChunk {

    override fun readByte(offset: Long): Byte {
        require(offset >= 0 && offset < size) { "Offset out of bounds: $offset" }
        return data[this.offset + offset.toInt()]
    }

    override fun readBytes(offset: Long, length: Int): ByteArray {
        require(offset >= 0 && offset + length <= size) { "Range out of bounds: $offset + $length" }
        return data.copyOfRange(this.offset + offset.toInt(), this.offset + offset.toInt() + length)
    }

    override fun slice(offset: Long, length: Long): MemoryChunk {
        require(offset >= 0 && offset + length <= size) { "Slice out of bounds: $offset + $length" }
        return ByteArrayMemoryChunk(data, this.offset + offset.toInt(), length)
    }
}
