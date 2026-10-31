package sk.ainet.lang.tensor

import sk.ainet.context.data
import sk.ainet.lang.tensor.dsl.DataContextDsl
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Tests for `Tensor.sliceCopy {}` (#1415): the slice is built like `sliceView {}` and
 * materialized through CopyMaterializationStrategy into an independent tensor.
 */
class SliceCopyTest {

    /** (4, 3) tensor whose value at (r, c) is its row-major flat index. */
    private fun DataContextDsl.sampleTensor(): Tensor<FP32, Float> =
        tensor<FP32, Float> {
            shape(4, 3) { shape ->
                init { indices ->
                    var flatIndex = 0
                    var stride = 1
                    for (i in indices.indices.reversed()) {
                        flatIndex += indices[i] * stride
                        stride *= shape[i]
                    }
                    flatIndex.toFloat()
                }
            }
        }

    @Test
    fun sliceCopy_returnsSlicedValuesWithSlicedShape() {
        data {
            val source = sampleTensor()

            val copy = source.sliceCopy {
                segment { range(1, 3) } // rows 1..2
                segment { all() }
            }

            assertEquals(Shape(2, 3), copy.shape)
            for (r in 0 until 2) {
                for (c in 0 until 3) {
                    assertEquals((r + 1) * 3 + c.toFloat(), copy.data[r, c], "value at ($r, $c)")
                }
            }
        }
    }

    @Test
    fun sliceCopy_atSlice_reducesRank() {
        data {
            val source = sampleTensor()

            val copy = source.sliceCopy {
                segment { at(2) } // row 2 -> rank drops to 1
                segment { all() }
            }

            assertEquals(Shape(3), copy.shape)
            for (c in 0 until 3) {
                assertEquals(6f + c, copy.data[c])
            }
        }
    }

    @Test
    fun sliceCopy_stepSlice_picksStridedValues() {
        data {
            val source = sampleTensor()

            val copy = source.sliceCopy {
                segment { step(0, 4, 2) } // rows 0 and 2
                segment { range(1, 3) } // columns 1..2
            }

            assertEquals(Shape(2, 2), copy.shape)
            assertEquals(1f, copy.data[0, 0])
            assertEquals(2f, copy.data[0, 1])
            assertEquals(7f, copy.data[1, 0])
            assertEquals(8f, copy.data[1, 1])
        }
    }

    @Test
    fun sliceCopy_isIndependent_mutatingCopyLeavesSourceUntouched() {
        data {
            val source = sampleTensor()

            val copy = source.sliceCopy {
                segment { range(0, 2) }
                segment { all() }
            }

            copy.data.set(0, 0, value = 1000f)
            copy.data.set(1, 2, value = -1000f)

            assertEquals(0f, source.data[0, 0], "source must not see writes to the copy")
            assertEquals(5f, source.data[1, 2], "source must not see writes to the copy")
            assertEquals(1000f, copy.data[0, 0])
            assertEquals(-1000f, copy.data[1, 2])
        }
    }

    @Test
    fun sliceCopy_isIndependent_mutatingSourceLeavesCopyUntouched() {
        data {
            val source = sampleTensor()

            val copy = source.sliceCopy {
                segment { range(0, 2) }
                segment { all() }
            }

            source.data.set(0, 0, value = 42f)

            assertEquals(0f, copy.data[0, 0], "copy must not see writes to the source (unlike sliceView)")
        }
    }
}
