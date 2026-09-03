package sk.ainet.exec.tensor.ops

import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.schedule.CoroutineSchedule

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder

/**
 * Thin wrapper over the JDK Vector API. Isolate incubator usage here so call sites stay clean
 * and future API changes are easier to adapt.
 */
internal object JvmVectorKernels {
    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    // region Elementwise
    fun binaryFloat(
        a: FloatArray,
        b: FloatArray,
        out: FloatArray,
        length: Int,
        op: (FloatVector, FloatVector) -> FloatVector,
        scalarOp: (Float, Float) -> Float,
        aOffset: Int = 0,
        bOffset: Int = 0,
    ) {
        var index = 0
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(length)
        while (index < loopBound) {
            val va = FloatVector.fromArray(floatSpecies, a, aOffset + index)
            val vb = FloatVector.fromArray(floatSpecies, b, bOffset + index)
            op(va, vb).intoArray(out, index)
            index += step
        }
        while (index < length) {
            out[index] = scalarOp(a[aOffset + index], b[bOffset + index])
            index++
        }
    }

    fun unaryFloat(
        input: FloatArray,
        out: FloatArray,
        length: Int,
        op: (FloatVector) -> FloatVector,
        scalarOp: (Float) -> Float,
        inputOffset: Int = 0,
    ) {
        var index = 0
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(length)
        while (index < loopBound) {
            val v = FloatVector.fromArray(floatSpecies, input, inputOffset + index)
            op(v).intoArray(out, index)
            index += step
        }
        while (index < length) {
            out[index] = scalarOp(input[inputOffset + index])
            index++
        }
    }
    // endregion

    // region Reductions
    fun reduceAllSumFloat(input: FloatArray, length: Int): Float {
        var index = 0
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(length)
        var accVec = FloatVector.zero(floatSpecies)
        while (index < loopBound) {
            val v = FloatVector.fromArray(floatSpecies, input, index)
            accVec = accVec.add(v)
            index += step
        }
        var acc = accVec.reduceLanes(VectorOperators.ADD)
        while (index < length) {
            acc += input[index]
            index++
        }
        return acc
    }
    // endregion

    // region Matmul (naive + vectorized inner product)
    fun matmulFloat(
        aRows: Int,
        aCols: Int,
        bCols: Int,
        a: FloatArray,
        b: FloatArray,
        out: FloatArray,
    ) {
        // b is expected in normal layout; we transpose internally for cache-friendly access
        val transposedB = FloatArray(bCols * aCols)
        // original b has dimensions aCols x bCols
        for (row in 0 until aCols) {
            val srcOffset = row * bCols
            for (col in 0 until bCols) {
                transposedB[col * aCols + row] = b[srcOffset + col]
            }
        }

        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(aCols)
        for (row in 0 until aRows) {
            val aOffset = row * aCols
            for (col in 0 until bCols) {
                val bOffset = col * aCols
                var idx = 0
                var accVec = FloatVector.zero(floatSpecies)
                while (idx < loopBound) {
                    val va = FloatVector.fromArray(floatSpecies, a, aOffset + idx)
                    val vb = FloatVector.fromArray(floatSpecies, transposedB, bOffset + idx)
                    accVec = accVec.add(va.mul(vb))
                    idx += step
                }
                var acc = accVec.reduceLanes(VectorOperators.ADD)
                while (idx < aCols) {
                    acc += a[aOffset + idx] * transposedB[bOffset + idx]
                    idx++
                }
                out[row * bCols + col] = acc
            }
        }
    }

    /**
     * Blocked (tiled) matmul with vectorized inner products. Designed for small/medium matrices.
     * Tiles of 8x8 generally perform well across x86/ARM with preferred species.
     */
    fun matmulFloatBlocked(
        aRows: Int,
        aCols: Int,
        bCols: Int,
        a: FloatArray,
        b: FloatArray,
        out: FloatArray,
        tileM: Int = 8,
        tileN: Int = 8,
        tileK: Int = 128,
    ) {
        // Transpose B for contiguous access along K
        val bt = FloatArray(bCols * aCols)
        for (k in 0 until aCols) {
            val src = k * bCols
            for (n in 0 until bCols) {
                bt[n * aCols + k] = b[src + n]
            }
        }
        val step = floatSpecies.length()
        val mBlocks = (aRows + tileM - 1) / tileM
        val nBlocks = (bCols + tileN - 1) / tileN
        val kBlocks = (aCols + tileK - 1) / tileK
        for (bm in 0 until mBlocks) {
            val mStart = bm * tileM
            val mEnd = minOf(mStart + tileM, aRows)
            for (bn in 0 until nBlocks) {
                val nStart = bn * tileN
                val nEnd = minOf(nStart + tileN, bCols)
                // Initialize C tile
                for (m in mStart until mEnd) {
                    val rowOff = m * bCols
                    for (n in nStart until nEnd) {
                        if (kBlocks == 0) {
                            out[rowOff + n] = 0f
                        } else if (bm == 0 && bn == 0) {
                            // ensure zeroing only once in simple flow; otherwise accumulate below
                            out[rowOff + n] = 0f
                        }
                    }
                }
                for (bk in 0 until kBlocks) {
                    val kStart = bk * tileK
                    val kEnd = minOf(kStart + tileK, aCols)
                    // Compute C[m, n] += A[m, k] * Bt[n, k] over kStart..kEnd
                    val loopBound = floatSpecies.loopBound(kEnd - kStart)
                    val kLen = kEnd - kStart
                    for (m in mStart until mEnd) {
                        val aBase = m * aCols + kStart
                        val cBase = m * bCols
                        for (n in nStart until nEnd) {
                            val btBase = n * aCols + kStart
                            var idx = 0
                            var accVec = FloatVector.zero(floatSpecies)
                            while (idx < loopBound) {
                                val va = FloatVector.fromArray(floatSpecies, a, aBase + idx)
                                val vb = FloatVector.fromArray(floatSpecies, bt, btBase + idx)
                                accVec = accVec.add(va.mul(vb))
                                idx += step
                            }
                            var acc = accVec.reduceLanes(VectorOperators.ADD)
                            while (idx < kLen) {
                                acc += a[aBase + idx] * bt[btBase + idx]
                                idx++
                            }
                            out[cBase + n] += acc
                        }
                    }
                }
            }
        }
    }
    // endregion

    // region Conv2d (im2col + vectorized matmul)

    /**
     * Optimized conv2d using im2col transformation + vectorized matmul.
     * Converts convolution into matrix multiplication for better cache utilization and SIMD.
     *
     * @param input Input tensor data in NCHW format, flattened
     * @param weight Weight tensor data in (C_out, C_in, kH, kW) format, flattened
     * @param bias Optional bias data of shape (C_out)
     * @param n Batch size
     * @param cIn Input channels
     * @param inH Input height
     * @param inW Input width
     * @param cOut Output channels
     * @param kH Kernel height
     * @param kW Kernel width
     * @param sH Stride height
     * @param sW Stride width
     * @param pH Padding height
     * @param pW Padding width
     * @param outH Output height
     * @param outW Output width
     * @param output Pre-allocated output buffer of size (n * cOut * outH * outW)
     */
    fun conv2dIm2Col(
        input: FloatArray,
        weight: FloatArray,
        bias: FloatArray?,
        n: Int, cIn: Int, inH: Int, inW: Int,
        cOut: Int, kH: Int, kW: Int,
        sH: Int, sW: Int, pH: Int, pW: Int,
        outH: Int, outW: Int,
        output: FloatArray
    ) {
        val colSize = cIn * kH * kW
        val patchCount = outH * outW

        // Process each batch
        for (batch in 0 until n) {
            val inputOffset = batch * cIn * inH * inW
            val outputOffset = batch * cOut * outH * outW

            // im2col: extract patches into column matrix
            // col shape: (patchCount, colSize) = (outH*outW, cIn*kH*kW)
            val col = FloatArray(patchCount * colSize)
            im2colNCHW(input, inputOffset, cIn, inH, inW, kH, kW, sH, sW, pH, pW, outH, outW, col)

            // weight shape: (cOut, cIn*kH*kW) - already in correct layout
            // matmul: col @ weight^T => (patchCount, cOut)
            // We compute: for each patch p, for each output channel oc:
            //   out[p, oc] = sum over k: col[p, k] * weight[oc, k]

            // Transpose weight for better memory access (weight is cOut x colSize)
            // After transpose: weightT is colSize x cOut
            // But actually we want col @ weightT which means: (patchCount x colSize) @ (colSize x cOut)
            // Output is (patchCount x cOut)

            // Actually for NCHW output we need (cOut, patchCount) then reshape
            // Let's compute weight @ col^T = (cOut, colSize) @ (colSize, patchCount) = (cOut, patchCount)

            matmulConv(weight, cOut, colSize, col, patchCount, output, outputOffset, bias)
        }
    }

    /**
     * Extract image patches into columns (im2col for NCHW format).
     */
    private fun im2colNCHW(
        input: FloatArray,
        inputOffset: Int,
        cIn: Int, inH: Int, inW: Int,
        kH: Int, kW: Int,
        sH: Int, sW: Int,
        pH: Int, pW: Int,
        outH: Int, outW: Int,
        col: FloatArray
    ) {
        val colSize = cIn * kH * kW
        var colIdx = 0

        for (oh in 0 until outH) {
            for (ow in 0 until outW) {
                val hStart = oh * sH - pH
                val wStart = ow * sW - pW

                for (c in 0 until cIn) {
                    val inputChannelOffset = inputOffset + c * inH * inW
                    for (kh in 0 until kH) {
                        val ih = hStart + kh
                        for (kw in 0 until kW) {
                            val iw = wStart + kw
                            col[colIdx++] = if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                input[inputChannelOffset + ih * inW + iw]
                            } else {
                                0f // Zero padding
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Vectorized matmul for conv2d: weight @ col^T + bias
     * weight: (cOut, colSize)
     * col: (patchCount, colSize) - we treat as colSize x patchCount for the matmul
     * output: (cOut, patchCount) written to output buffer
     */
    private fun matmulConv(
        weight: FloatArray,
        cOut: Int,
        colSize: Int,
        col: FloatArray,
        patchCount: Int,
        output: FloatArray,
        outputOffset: Int,
        bias: FloatArray?
    ) {
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(colSize)

        for (oc in 0 until cOut) {
            val weightRowOffset = oc * colSize
            val biasVal = bias?.get(oc) ?: 0f

            for (p in 0 until patchCount) {
                // Compute dot product: weight[oc, :] · col[p, :]
                var idx = 0
                var accVec = FloatVector.zero(floatSpecies)

                while (idx < loopBound) {
                    val vw = FloatVector.fromArray(floatSpecies, weight, weightRowOffset + idx)
                    val vc = FloatVector.fromArray(floatSpecies, col, p * colSize + idx)
                    accVec = accVec.add(vw.mul(vc))
                    idx += step
                }

                var acc = accVec.reduceLanes(VectorOperators.ADD)

                // Scalar tail
                while (idx < colSize) {
                    acc += weight[weightRowOffset + idx] * col[p * colSize + idx]
                    idx++
                }

                output[outputOffset + oc * patchCount + p] = acc + biasVal
            }
        }
    }

    /**
     * Direct vectorized conv2d without im2col for small kernels.
     * More memory efficient for 1x1 convolutions.
     */
    fun conv2dDirect1x1(
        input: FloatArray,
        weight: FloatArray,
        bias: FloatArray?,
        n: Int, cIn: Int, inH: Int, inW: Int,
        cOut: Int,
        output: FloatArray
    ) {
        val spatialSize = inH * inW
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(cIn)

        for (batch in 0 until n) {
            val inputBatchOffset = batch * cIn * spatialSize
            val outputBatchOffset = batch * cOut * spatialSize

            for (oc in 0 until cOut) {
                val weightRowOffset = oc * cIn
                val biasVal = bias?.get(oc) ?: 0f
                val outputChannelOffset = outputBatchOffset + oc * spatialSize

                for (sp in 0 until spatialSize) {
                    var idx = 0
                    var accVec = FloatVector.zero(floatSpecies)

                    while (idx < loopBound) {
                        val vw = FloatVector.fromArray(floatSpecies, weight, weightRowOffset + idx)
                        // Gather input values for this spatial position across channels
                        // input[batch, ic, h, w] = input[inputBatchOffset + ic * spatialSize + sp]
                        // This is strided access - need to handle differently

                        // For 1x1 conv, we can reorganize: instead of looping over channels with SIMD,
                        // we can loop over spatial positions with SIMD
                        idx += step
                    }

                    // Fall back to scalar for strided channel access
                    var acc = 0f
                    for (ic in 0 until cIn) {
                        acc += weight[weightRowOffset + ic] * input[inputBatchOffset + ic * spatialSize + sp]
                    }
                    output[outputChannelOffset + sp] = acc + biasVal
                }
            }
        }
    }

    /**
     * Optimized 1x1 conv with channel-last reordering for better SIMD utilization.
     */
    fun conv2d1x1Optimized(
        input: FloatArray,
        weight: FloatArray,
        bias: FloatArray?,
        n: Int, cIn: Int, inH: Int, inW: Int,
        cOut: Int,
        output: FloatArray
    ) {
        val spatialSize = inH * inW
        val step = floatSpecies.length()

        for (batch in 0 until n) {
            val inputBatchOffset = batch * cIn * spatialSize
            val outputBatchOffset = batch * cOut * spatialSize

            // Reorder input to channel-last: (spatial, cIn)
            val inputReordered = FloatArray(spatialSize * cIn)
            for (sp in 0 until spatialSize) {
                for (ic in 0 until cIn) {
                    inputReordered[sp * cIn + ic] = input[inputBatchOffset + ic * spatialSize + sp]
                }
            }

            // Now we can do efficient matmul: inputReordered @ weight^T
            // inputReordered: (spatialSize, cIn)
            // weight: (cOut, cIn)
            // output: (spatialSize, cOut)

            val loopBound = floatSpecies.loopBound(cIn)

            for (sp in 0 until spatialSize) {
                val inputRowOffset = sp * cIn

                for (oc in 0 until cOut) {
                    val weightRowOffset = oc * cIn
                    val biasVal = bias?.get(oc) ?: 0f

                    var idx = 0
                    var accVec = FloatVector.zero(floatSpecies)

                    while (idx < loopBound) {
                        val vi = FloatVector.fromArray(floatSpecies, inputReordered, inputRowOffset + idx)
                        val vw = FloatVector.fromArray(floatSpecies, weight, weightRowOffset + idx)
                        accVec = accVec.add(vi.mul(vw))
                        idx += step
                    }

                    var acc = accVec.reduceLanes(VectorOperators.ADD)
                    while (idx < cIn) {
                        acc += inputReordered[inputRowOffset + idx] * weight[weightRowOffset + idx]
                        idx++
                    }

                    // Write to output in NCHW format
                    output[outputBatchOffset + oc * spatialSize + sp] = acc + biasVal
                }
            }
        }
    }
    // endregion

    // region Conv1d (im2col + vectorized matmul)

    /**
     * Optimized conv1d using im2col transformation + vectorized matmul.
     */
    fun conv1dIm2Col(
        input: FloatArray,
        weight: FloatArray,
        bias: FloatArray?,
        n: Int, cIn: Int, inL: Int,
        cOut: Int, kL: Int,
        stride: Int, padding: Int, dilation: Int,
        outL: Int,
        output: FloatArray
    ) {
        val colSize = cIn * kL
        val patchCount = outL

        for (batch in 0 until n) {
            val inputOffset = batch * cIn * inL
            val outputOffset = batch * cOut * outL

            // im2col for 1D
            val col = FloatArray(patchCount * colSize)
            im2col1D(input, inputOffset, cIn, inL, kL, stride, padding, dilation, outL, col)

            // matmul: weight @ col^T
            matmulConv1d(weight, cOut, colSize, col, patchCount, output, outputOffset, bias)
        }
    }

    private fun im2col1D(
        input: FloatArray,
        inputOffset: Int,
        cIn: Int, inL: Int,
        kL: Int,
        stride: Int, padding: Int, dilation: Int,
        outL: Int,
        col: FloatArray
    ) {
        var colIdx = 0
        for (ol in 0 until outL) {
            val lStart = ol * stride - padding
            for (c in 0 until cIn) {
                val inputChannelOffset = inputOffset + c * inL
                for (kl in 0 until kL) {
                    val il = lStart + kl * dilation
                    col[colIdx++] = if (il >= 0 && il < inL) {
                        input[inputChannelOffset + il]
                    } else {
                        0f
                    }
                }
            }
        }
    }

    private fun matmulConv1d(
        weight: FloatArray,
        cOut: Int,
        colSize: Int,
        col: FloatArray,
        patchCount: Int,
        output: FloatArray,
        outputOffset: Int,
        bias: FloatArray?
    ) {
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(colSize)

        for (oc in 0 until cOut) {
            val weightRowOffset = oc * colSize
            val biasVal = bias?.get(oc) ?: 0f

            for (p in 0 until patchCount) {
                var idx = 0
                var accVec = FloatVector.zero(floatSpecies)

                while (idx < loopBound) {
                    val vw = FloatVector.fromArray(floatSpecies, weight, weightRowOffset + idx)
                    val vc = FloatVector.fromArray(floatSpecies, col, p * colSize + idx)
                    accVec = accVec.add(vw.mul(vc))
                    idx += step
                }

                var acc = accVec.reduceLanes(VectorOperators.ADD)
                while (idx < colSize) {
                    acc += weight[weightRowOffset + idx] * col[p * colSize + idx]
                    idx++
                }

                output[outputOffset + oc * patchCount + p] = acc + biasVal
            }
        }
    }
    // endregion

    // region Conv2d with dilation support

    /**
     * Conv2d with dilation support using im2col.
     */
    fun conv2dIm2ColDilated(
        input: FloatArray,
        weight: FloatArray,
        bias: FloatArray?,
        n: Int, cIn: Int, inH: Int, inW: Int,
        cOut: Int, kH: Int, kW: Int,
        sH: Int, sW: Int, pH: Int, pW: Int,
        dH: Int, dW: Int,
        outH: Int, outW: Int,
        output: FloatArray
    ) {
        val colSize = cIn * kH * kW
        val patchCount = outH * outW

        for (batch in 0 until n) {
            val inputOffset = batch * cIn * inH * inW
            val outputOffset = batch * cOut * outH * outW

            val col = FloatArray(patchCount * colSize)
            im2colNCHWDilated(input, inputOffset, cIn, inH, inW, kH, kW, sH, sW, pH, pW, dH, dW, outH, outW, col)
            matmulConv(weight, cOut, colSize, col, patchCount, output, outputOffset, bias)
        }
    }

    private fun im2colNCHWDilated(
        input: FloatArray,
        inputOffset: Int,
        cIn: Int, inH: Int, inW: Int,
        kH: Int, kW: Int,
        sH: Int, sW: Int,
        pH: Int, pW: Int,
        dH: Int, dW: Int,
        outH: Int, outW: Int,
        col: FloatArray
    ) {
        var colIdx = 0
        for (oh in 0 until outH) {
            for (ow in 0 until outW) {
                val hStart = oh * sH - pH
                val wStart = ow * sW - pW

                for (c in 0 until cIn) {
                    val inputChannelOffset = inputOffset + c * inH * inW
                    for (kh in 0 until kH) {
                        val ih = hStart + kh * dH
                        for (kw in 0 until kW) {
                            val iw = wStart + kw * dW
                            col[colIdx++] = if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                input[inputChannelOffset + ih * inW + iw]
                            } else {
                                0f
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Grouped conv2d using im2col per group.
     */
    fun conv2dGrouped(
        input: FloatArray,
        weight: FloatArray,
        bias: FloatArray?,
        n: Int, cIn: Int, inH: Int, inW: Int,
        cOut: Int, kH: Int, kW: Int,
        sH: Int, sW: Int, pH: Int, pW: Int,
        dH: Int, dW: Int,
        groups: Int,
        outH: Int, outW: Int,
        output: FloatArray
    ) {
        val cInPerGroup = cIn / groups
        val cOutPerGroup = cOut / groups
        val colSize = cInPerGroup * kH * kW
        val patchCount = outH * outW

        for (batch in 0 until n) {
            for (g in 0 until groups) {
                val inputGroupOffset = batch * cIn * inH * inW + g * cInPerGroup * inH * inW
                val outputGroupOffset = batch * cOut * outH * outW + g * cOutPerGroup * outH * outW
                val weightGroupOffset = g * cOutPerGroup * colSize

                // im2col for this group's input channels
                val col = FloatArray(patchCount * colSize)
                im2colNCHWGrouped(input, inputGroupOffset, cInPerGroup, inH, inW, kH, kW, sH, sW, pH, pW, dH, dW, outH, outW, col)

                // matmul for this group
                matmulConvGrouped(weight, weightGroupOffset, cOutPerGroup, colSize, col, patchCount, output, outputGroupOffset, bias, g * cOutPerGroup)
            }
        }
    }

    private fun im2colNCHWGrouped(
        input: FloatArray,
        inputOffset: Int,
        cInPerGroup: Int, inH: Int, inW: Int,
        kH: Int, kW: Int,
        sH: Int, sW: Int,
        pH: Int, pW: Int,
        dH: Int, dW: Int,
        outH: Int, outW: Int,
        col: FloatArray
    ) {
        var colIdx = 0
        for (oh in 0 until outH) {
            for (ow in 0 until outW) {
                val hStart = oh * sH - pH
                val wStart = ow * sW - pW

                for (c in 0 until cInPerGroup) {
                    val inputChannelOffset = inputOffset + c * inH * inW
                    for (kh in 0 until kH) {
                        val ih = hStart + kh * dH
                        for (kw in 0 until kW) {
                            val iw = wStart + kw * dW
                            col[colIdx++] = if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                input[inputChannelOffset + ih * inW + iw]
                            } else {
                                0f
                            }
                        }
                    }
                }
            }
        }
    }

    private fun matmulConvGrouped(
        weight: FloatArray,
        weightOffset: Int,
        cOutPerGroup: Int,
        colSize: Int,
        col: FloatArray,
        patchCount: Int,
        output: FloatArray,
        outputOffset: Int,
        bias: FloatArray?,
        biasOffset: Int
    ) {
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(colSize)

        for (oc in 0 until cOutPerGroup) {
            val weightRowOffset = weightOffset + oc * colSize
            val biasVal = bias?.get(biasOffset + oc) ?: 0f

            for (p in 0 until patchCount) {
                var idx = 0
                var accVec = FloatVector.zero(floatSpecies)

                while (idx < loopBound) {
                    val vw = FloatVector.fromArray(floatSpecies, weight, weightRowOffset + idx)
                    val vc = FloatVector.fromArray(floatSpecies, col, p * colSize + idx)
                    accVec = accVec.add(vw.mul(vc))
                    idx += step
                }

                var acc = accVec.reduceLanes(VectorOperators.ADD)
                while (idx < colSize) {
                    acc += weight[weightRowOffset + idx] * col[p * colSize + idx]
                    idx++
                }

                output[outputOffset + oc * patchCount + p] = acc + biasVal
            }
        }
    }
    // endregion

    // region Conv3d (im2col + vectorized matmul)

    /**
     * Optimized conv3d using im2col transformation + vectorized matmul.
     */
    fun conv3dIm2Col(
        input: FloatArray,
        weight: FloatArray,
        bias: FloatArray?,
        n: Int, cIn: Int, inD: Int, inH: Int, inW: Int,
        cOut: Int, kD: Int, kH: Int, kW: Int,
        sD: Int, sH: Int, sW: Int,
        pD: Int, pH: Int, pW: Int,
        dD: Int, dH: Int, dW: Int,
        outD: Int, outH: Int, outW: Int,
        output: FloatArray
    ) {
        val colSize = cIn * kD * kH * kW
        val patchCount = outD * outH * outW

        for (batch in 0 until n) {
            val inputOffset = batch * cIn * inD * inH * inW
            val outputOffset = batch * cOut * outD * outH * outW

            val col = FloatArray(patchCount * colSize)
            im2col3D(input, inputOffset, cIn, inD, inH, inW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, outD, outH, outW, col)
            matmulConv3d(weight, cOut, colSize, col, patchCount, output, outputOffset, bias)
        }
    }

    private fun im2col3D(
        input: FloatArray,
        inputOffset: Int,
        cIn: Int, inD: Int, inH: Int, inW: Int,
        kD: Int, kH: Int, kW: Int,
        sD: Int, sH: Int, sW: Int,
        pD: Int, pH: Int, pW: Int,
        dD: Int, dH: Int, dW: Int,
        outD: Int, outH: Int, outW: Int,
        col: FloatArray
    ) {
        var colIdx = 0
        for (od in 0 until outD) {
            for (oh in 0 until outH) {
                for (ow in 0 until outW) {
                    val dStart = od * sD - pD
                    val hStart = oh * sH - pH
                    val wStart = ow * sW - pW

                    for (c in 0 until cIn) {
                        val inputChannelOffset = inputOffset + c * inD * inH * inW
                        for (kd in 0 until kD) {
                            val id = dStart + kd * dD
                            for (kh in 0 until kH) {
                                val ih = hStart + kh * dH
                                for (kw in 0 until kW) {
                                    val iw = wStart + kw * dW
                                    col[colIdx++] = if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                        input[inputChannelOffset + id * inH * inW + ih * inW + iw]
                                    } else {
                                        0f
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private fun matmulConv3d(
        weight: FloatArray,
        cOut: Int,
        colSize: Int,
        col: FloatArray,
        patchCount: Int,
        output: FloatArray,
        outputOffset: Int,
        bias: FloatArray?
    ) {
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(colSize)

        for (oc in 0 until cOut) {
            val weightRowOffset = oc * colSize
            val biasVal = bias?.get(oc) ?: 0f

            for (p in 0 until patchCount) {
                var idx = 0
                var accVec = FloatVector.zero(floatSpecies)

                while (idx < loopBound) {
                    val vw = FloatVector.fromArray(floatSpecies, weight, weightRowOffset + idx)
                    val vc = FloatVector.fromArray(floatSpecies, col, p * colSize + idx)
                    accVec = accVec.add(vw.mul(vc))
                    idx += step
                }

                var acc = accVec.reduceLanes(VectorOperators.ADD)
                while (idx < colSize) {
                    acc += weight[weightRowOffset + idx] * col[p * colSize + idx]
                    idx++
                }

                output[outputOffset + oc * patchCount + p] = acc + biasVal
            }
        }
    }
    // endregion

    // region MemorySegment-based matmul

    private val BYTE_ORDER = ByteOrder.LITTLE_ENDIAN

    /**
     * Vectorized matmul for MemorySegment-backed tensors.
     * Uses FloatVector.fromMemorySegment for SIMD-friendly off-heap access.
     *
     * @param m number of rows in A
     * @param k inner dimension (cols of A = rows of B)
     * @param n number of cols in B
     * @param aSeg MemorySegment for matrix A
     * @param aByteOffset byte offset into aSeg
     * @param bSeg MemorySegment for matrix B
     * @param bByteOffset byte offset into bSeg
     * @param rSeg MemorySegment for result matrix
     * @param rByteOffset byte offset into rSeg
     */
    fun matmulFloatMemSeg(
        m: Int, k: Int, n: Int,
        aSeg: MemorySegment, aByteOffset: Long,
        bSeg: MemorySegment, bByteOffset: Long,
        rSeg: MemorySegment, rByteOffset: Long,
    ) {
        // Transpose B into a temporary FloatArray for contiguous access along K
        val bt = FloatArray(n * k)
        for (row in 0 until k) {
            val srcByteOff = bByteOffset + row.toLong() * n * Float.SIZE_BYTES
            for (col in 0 until n) {
                bt[col * k + row] = bSeg.get(
                    java.lang.foreign.ValueLayout.JAVA_FLOAT.withOrder(BYTE_ORDER),
                    srcByteOff + col.toLong() * Float.SIZE_BYTES,
                )
            }
        }

        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(k)
        val floatBytes = Float.SIZE_BYTES.toLong()

        for (row in 0 until m) {
            val aRowByteOff = aByteOffset + row.toLong() * k * floatBytes
            for (col in 0 until n) {
                val btOff = col * k
                var idx = 0
                var accVec = FloatVector.zero(floatSpecies)
                while (idx < loopBound) {
                    val va = FloatVector.fromMemorySegment(
                        floatSpecies, aSeg, aRowByteOff + idx.toLong() * floatBytes, BYTE_ORDER,
                    )
                    val vb = FloatVector.fromArray(floatSpecies, bt, btOff + idx)
                    accVec = va.fma(vb, accVec)
                    idx += step
                }
                var acc = accVec.reduceLanes(VectorOperators.ADD)
                while (idx < k) {
                    acc += aSeg.get(
                        java.lang.foreign.ValueLayout.JAVA_FLOAT.withOrder(BYTE_ORDER),
                        aRowByteOff + idx.toLong() * floatBytes,
                    ) * bt[btOff + idx]
                    idx++
                }
                rSeg.set(
                    java.lang.foreign.ValueLayout.JAVA_FLOAT.withOrder(BYTE_ORDER),
                    rByteOffset + (row.toLong() * n + col) * floatBytes,
                    acc,
                )
            }
        }
    }

    /**
     * Blocked (tiled) matmul for MemorySegment-backed tensors.
     * Better cache utilization for larger matrices.
     */
    fun matmulFloatBlockedMemSeg(
        m: Int, k: Int, n: Int,
        aSeg: MemorySegment, aByteOffset: Long,
        bSeg: MemorySegment, bByteOffset: Long,
        rSeg: MemorySegment, rByteOffset: Long,
        tileM: Int = 8,
        tileN: Int = 8,
        tileK: Int = 128,
            schedule: Schedule = CoroutineSchedule.hardware(),
    ) {
        val floatLayout = java.lang.foreign.ValueLayout.JAVA_FLOAT.withOrder(BYTE_ORDER)
        val floatBytes = Float.SIZE_BYTES.toLong()

        // Bulk-load A and B from MemorySegment into FloatArrays. Per-element
        // VarHandle.get is O(m*k + n*k) and dominates the matmul wall time
        // for attention (QK^T, AV) where both operands are MemSeg-backed.
        // MemorySegment.copy(seg, layout, off, array, ...) issues a single
        // native memcopy per row.
        val a = FloatArray(m * k)
        MemorySegment.copy(aSeg, floatLayout, aByteOffset, a, 0, m * k)

        // Transpose B (row-major n*k → column-major bt[nn * k + kk]) using
        // bulk row reads + scalar scatter (the scatter is a tight write loop
        // the JIT auto-vectorizes).
        val bt = FloatArray(n * k)
        val rowBuf = FloatArray(n)
        for (kk in 0 until k) {
            val srcByteOff = bByteOffset + kk.toLong() * n * floatBytes
            MemorySegment.copy(bSeg, floatLayout, srcByteOff, rowBuf, 0, n)
            for (nn in 0 until n) {
                bt[nn * k + kk] = rowBuf[nn]
            }
        }

        // Local accumulator — one write per (mm, nn) at the end.
        val r = FloatArray(m * n)

        val step = floatSpecies.length()
        val nBlocks = (n + tileN - 1) / tileN
        val kBlocks = (k + tileK - 1) / tileK

        // Parallelize over m (independent rows of the result). Each task owns
        // a contiguous mm range and writes to its own slice of `r`. Tiling on
        // n and k stays for cache locality.
        parallelChunks(m, schedule) { mStart, mEnd ->
            for (bn in 0 until nBlocks) {
                val nStart = bn * tileN
                val nEnd = minOf(nStart + tileN, n)
                for (bk in 0 until kBlocks) {
                    val kStart = bk * tileK
                    val kEnd = minOf(kStart + tileK, k)
                    val kLen = kEnd - kStart
                    val loopBound = floatSpecies.loopBound(kLen)

                    for (mm in mStart until mEnd) {
                        val aBase = mm * k + kStart
                        for (nn in nStart until nEnd) {
                            val btBase = nn * k + kStart
                            var idx = 0
                            var accVec = FloatVector.zero(floatSpecies)
                            while (idx < loopBound) {
                                val va = FloatVector.fromArray(floatSpecies, a, aBase + idx)
                                val vb = FloatVector.fromArray(floatSpecies, bt, btBase + idx)
                                accVec = va.fma(vb, accVec)
                                idx += step
                            }
                            var acc = accVec.reduceLanes(VectorOperators.ADD)
                            while (idx < kLen) {
                                acc += a[aBase + idx] * bt[btBase + idx]
                                idx++
                            }
                            r[mm * n + nn] += acc
                        }
                    }
                }
            }
        }

        // Bulk-write the result back to MemSeg in one call.
        MemorySegment.copy(r, 0, rSeg, floatLayout, rByteOffset, m * n)
    }

    /**
     * Batched dot-product for attention score computation on MemorySegment data.
     * Computes dotProduct(query, key[t]) for t in 0..currentPos.
     */
    fun batchDotProductMemSeg(
        query: MemorySegment, queryByteOffset: Long,
        keys: MemorySegment, keysByteOffset: Long,
        headSize: Int,
        keyStride: Int,
        positions: Int,
        scale: Float,
        output: FloatArray,
    ) {
        val floatLayout = java.lang.foreign.ValueLayout.JAVA_FLOAT.withOrder(BYTE_ORDER)
        val floatBytes = Float.SIZE_BYTES.toLong()
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(headSize)

        for (t in 0 until positions) {
            val keyOff = keysByteOffset + t.toLong() * keyStride * floatBytes
            var idx = 0
            var accVec = FloatVector.zero(floatSpecies)
            while (idx < loopBound) {
                val vq = FloatVector.fromMemorySegment(
                    floatSpecies, query, queryByteOffset + idx.toLong() * floatBytes, BYTE_ORDER,
                )
                val vk = FloatVector.fromMemorySegment(
                    floatSpecies, keys, keyOff + idx.toLong() * floatBytes, BYTE_ORDER,
                )
                accVec = vq.fma(vk, accVec)
                idx += step
            }
            var acc = accVec.reduceLanes(VectorOperators.ADD)
            while (idx < headSize) {
                acc += query.get(floatLayout, queryByteOffset + idx.toLong() * floatBytes) *
                    keys.get(floatLayout, keyOff + idx.toLong() * floatBytes)
                idx++
            }
            output[t] = acc * scale
        }
    }
    // endregion
}
