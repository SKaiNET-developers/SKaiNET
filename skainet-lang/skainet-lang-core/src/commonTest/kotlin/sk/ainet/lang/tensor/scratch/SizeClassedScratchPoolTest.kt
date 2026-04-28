package sk.ainet.lang.tensor.scratch

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotSame
import kotlin.test.assertSame
import kotlin.test.assertTrue

class SizeClassedScratchPoolTest {

    @Test
    fun sizeClassRoundsToNextPowerOfTwo() {
        assertEquals(0, SizeClassedScratchPool.sizeClass(1))
        assertEquals(0, SizeClassedScratchPool.sizeClass(64))
        assertEquals(1, SizeClassedScratchPool.sizeClass(65))
        assertEquals(1, SizeClassedScratchPool.sizeClass(128))
        assertEquals(2, SizeClassedScratchPool.sizeClass(129))
        assertEquals(2, SizeClassedScratchPool.sizeClass(256))
        assertEquals(3, SizeClassedScratchPool.sizeClass(257))
    }

    @Test
    fun sizeForClassReturnsBucketCapacity() {
        assertEquals(64, SizeClassedScratchPool.sizeForClass(0))
        assertEquals(128, SizeClassedScratchPool.sizeForClass(1))
        assertEquals(256, SizeClassedScratchPool.sizeForClass(2))
    }

    @Test
    fun bufferIsAtLeastRequestedSize() {
        val pool = SizeClassedScratchPool()
        pool.scope {
            assertTrue(pool.acquireFloat(1).size >= 1)
            assertTrue(pool.acquireFloat(100).size >= 100)
            assertTrue(pool.acquireFloat(1000).size >= 1000)
        }
    }

    @Test
    fun scopeRecyclesBuffers() {
        val pool = SizeClassedScratchPool()
        val first: FloatArray = pool.scope { pool.acquireFloat(100) }
        val second: FloatArray = pool.scope { pool.acquireFloat(100) }
        // Same bucket, recycled buffer comes back.
        assertSame(first, second)
    }

    @Test
    fun differentBucketsDoNotShare() {
        val pool = SizeClassedScratchPool()
        val small: FloatArray = pool.scope { pool.acquireFloat(50) }
        val large: FloatArray = pool.scope { pool.acquireFloat(500) }
        assertNotSame(small, large)
    }

    @Test
    fun acquireZeroedClearsRequestedRange() {
        val pool = SizeClassedScratchPool()
        pool.scope {
            val buf = pool.acquireFloat(100)
            for (i in 0 until 100) buf[i] = 7f
        }
        pool.scope {
            val buf = pool.acquireFloatZeroed(100)
            for (i in 0 until 100) assertEquals(0f, buf[i])
        }
    }

    @Test
    fun acquireOutsideScopeStillAllocates() {
        val pool = SizeClassedScratchPool()
        val buf = pool.acquireFloat(10)
        assertTrue(buf.size >= 10)
    }

    @Test
    fun nestedScopesBalance() {
        val pool = SizeClassedScratchPool()
        var innerRef: FloatArray? = null
        pool.scope {
            val outer = pool.acquireFloat(100)
            pool.scope {
                innerRef = pool.acquireFloat(100)
                assertNotSame(outer, innerRef)
            }
            // Inner-scope buffer was recycled on inner-scope exit — the next
            // acquire in the outer scope at the same size class returns it.
            val recycled = pool.acquireFloat(100)
            assertSame(innerRef, recycled)
        }
    }

    @Test
    fun statsReportAcquiresAndHits() {
        val pool = SizeClassedScratchPool()
        pool.scope { pool.acquireFloat(100) }
        pool.scope { pool.acquireFloat(100) } // hit
        pool.scope { pool.acquireFloat(100) } // hit
        val stats = pool.stats()
        assertEquals(3L, stats.acquireCount)
        assertEquals(2L, stats.cacheHits)
    }

    @Test
    fun surplusBuffersDropped() {
        val pool = SizeClassedScratchPool(maxBuffersPerClass = 2)
        // Acquire 3 buffers in one scope; on exit, only 2 are retained
        // (the third is dropped because the per-class cap is 2).
        pool.scope {
            pool.acquireFloat(100)
            pool.acquireFloat(100)
            pool.acquireFloat(100)
        }
        // Acquire 3 buffers in a second scope. First 2 hit the cache; the
        // 3rd misses because the cache only retained 2 from the prior scope.
        pool.scope {
            pool.acquireFloat(100)
            pool.acquireFloat(100)
            pool.acquireFloat(100)
        }
        val stats = pool.stats()
        assertEquals(6L, stats.acquireCount)
        assertEquals(2L, stats.cacheHits)
    }

    @Test
    fun noopPoolAlwaysAllocates() {
        val pool = NoopScratchPool
        val first = pool.scope { pool.acquireFloat(100) }
        val second = pool.scope { pool.acquireFloat(100) }
        assertNotSame(first, second)
    }
}
